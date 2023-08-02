import numpy as np
import torch
import torch.nn as nn

from nets.backbone import Backbone, C2f, Conv
from nets.yolo_training import weights_init
from utils.utils_bbox import make_anchors

def fuse_conv_and_bn(conv, bn):
    # 混合Conv2d + BatchNorm2d 减少计算量
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备kernel
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

class DFL(nn.Module):
    # DFL模块
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv   = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x           = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1     = c1

    def forward(self, x):
        # bs, self.reg_max * 4, 8400
        b, c, a = x.shape
        # bs, 4, self.reg_max, 8400 => bs, self.reg_max, 4, 8400 => b, 4, 8400
        # 以softmax的方式，对0~16的数字计算百分比，获得最终数字。
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)
        
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, input_shape, num_classes, phi, pretrained=False,task=['roadsign_voc','sagittaria_v1_voc']):
        super(YoloBody, self).__init__()

        self.task_number = len(num_classes)
        self.task_name = task


        depth_dict          = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
        width_dict          = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        deep_width_dict     = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        #-----------------------------------------------#
        #   输入图片是3, 640, 640
        #-----------------------------------------------#

        #---------------------------------------------------#   
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   256, 80, 80
        #   512, 40, 40
        #   1024 * deep_mul, 20, 20
        #---------------------------------------------------#
        self.backbone   = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)

        #------------------------加强特征提取网络------------------------# 
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        self.conv3_for_upsample1    = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        # 768, 80, 80 => 256, 80, 80
        self.conv3_for_upsample2    = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth, shortcut=False)
        
        # 256, 80, 80 => 256, 40, 40
        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        # 512 + 256, 40, 40 => 512, 40, 40
        self.conv3_for_downsample1  = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth, shortcut=False)

        # 512, 40, 40 => 512, 20, 20
        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        # 1024 * deep_mul + 512, 20, 20 =>  1024 * deep_mul, 20, 20
        self.conv3_for_downsample2  = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
        #------------------------加强特征提取网络------------------------# 
        
        ch              = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape      = None
        self.nl         = len(ch)
        # self.stride     = torch.zeros(self.nl)
        self.stride     = torch.tensor([256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])  # forward
        self.reg_max    = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no         = [i + self.reg_max * 4 for i in num_classes] # number of outputs per anchor
        self.num_classes = num_classes
        
        # c2, c3   = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)  # channels

        if self.task_number>=1:
            c2_1, c3_1 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes[0])  # channels
            self.cv2_1 = nn.ModuleList(
                nn.Sequential(Conv(x, c2_1, 3), Conv(c2_1, c2_1, 3), nn.Conv2d(c2_1, 4 * self.reg_max, 1)) for x in ch)
            self.cv3_1 = nn.ModuleList(
                nn.Sequential(Conv(x, c3_1, 3), Conv(c3_1, c3_1, 3), nn.Conv2d(c3_1, num_classes[0], 1)) for x in ch)
            self.dfl_1 = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.task_number>=2:
            c2_2, c3_2 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes[1])  # channels
            self.cv2_2 = nn.ModuleList(
                nn.Sequential(Conv(x, c2_2, 3), Conv(c2_2, c2_2, 3), nn.Conv2d(c2_2, 4 * self.reg_max, 1)) for x in ch)
            self.cv3_2 = nn.ModuleList(
                nn.Sequential(Conv(x, c3_2, 3), Conv(c3_2, c3_2, 3), nn.Conv2d(c3_2, num_classes[1], 1)) for x in ch)
            self.dfl_2 = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.task_number>=3:
            c2_3, c3_3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes[2])  # channels
            self.cv2_3 = nn.ModuleList(
                nn.Sequential(Conv(x, c2_3, 3), Conv(c2_3, c2_3, 3), nn.Conv2d(c2_3, 4 * self.reg_max, 1)) for x in ch)
            self.cv3_3 = nn.ModuleList(
                nn.Sequential(Conv(x, c3_3, 3), Conv(c3_3, c3_3, 3), nn.Conv2d(c3_3, num_classes[2], 1)) for x in ch)
            self.dfl_3 = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.task_number>=4:
            c2_4, c3_4 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes[3])  # channels
            self.cv2_4 = nn.ModuleList(
                nn.Sequential(Conv(x, c2_4, 3), Conv(c2_4, c2_4, 3), nn.Conv2d(c2_4, 4 * self.reg_max, 1)) for x in ch)
            self.cv3_4 = nn.ModuleList(
                nn.Sequential(Conv(x, c3_4, 3), Conv(c3_4, c3_4, 3), nn.Conv2d(c3_4, num_classes[3], 1)) for x in ch)
            self.dfl_4 = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()


        if self.task_number>=5:
            c2_5, c3_5 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes[4])  # channels
            self.cv2_5 = nn.ModuleList(
                nn.Sequential(Conv(x, c2_5, 3), Conv(c2_5, c2_5, 3), nn.Conv2d(c2_5, 4 * self.reg_max, 1)) for x in ch)
            self.cv3_5 = nn.ModuleList(
                nn.Sequential(Conv(x, c3_5, 3), Conv(c3_5, c3_5, 3), nn.Conv2d(c3_5, num_classes[4], 1)) for x in ch)
            self.dfl_5 = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()


        if not pretrained:
            weights_init(self)

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self
    
    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)
        
        #------------------------加强特征提取网络------------------------# 
        # 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 40, 40
        P5_upsample = self.upsample(feat3)
        # 1024 * deep_mul, 40, 40 cat 512, 40, 40 => 1024 * deep_mul + 512, 40, 40
        P4          = torch.cat([P5_upsample, feat2], 1)
        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        P4          = self.conv3_for_upsample1(P4)

        # 512, 40, 40 => 512, 80, 80
        P4_upsample = self.upsample(P4)
        # 512, 80, 80 cat 256, 80, 80 => 768, 80, 80
        P3          = torch.cat([P4_upsample, feat1], 1)
        # 768, 80, 80 => 256, 80, 80
        P3          = self.conv3_for_upsample2(P3)

        # 256, 80, 80 => 256, 40, 40
        P3_downsample = self.down_sample1(P3)
        # 512, 40, 40 cat 256, 40, 40 => 768, 40, 40
        P4 = torch.cat([P3_downsample, P4], 1)
        # 768, 40, 40 => 512, 40, 40
        P4 = self.conv3_for_downsample1(P4)

        # 512, 40, 40 => 512, 20, 20
        P4_downsample = self.down_sample2(P4)
        # 512, 20, 20 cat 1024 * deep_mul, 20, 20 => 1024 * deep_mul + 512, 20, 20
        P5 = torch.cat([P4_downsample, feat3], 1)
        # 1024 * deep_mul + 512, 20, 20 => 1024 * deep_mul, 20, 20
        P5 = self.conv3_for_downsample2(P5)
        #------------------------加强特征提取网络------------------------# 
        # P3 256, 80, 80
        # P4 512, 40, 40
        # P5 1024 * deep_mul, 20, 20
        shape = P3.shape  # BCHW
        
        # P3 256, 80, 80 => num_classes + self.reg_max * 4, 80, 80
        # P4 512, 40, 40 => num_classes + self.reg_max * 4, 40, 40
        # P5 1024 * deep_mul, 20, 20 => num_classes + self.reg_max * 4, 20, 20


        if self.task_number >= 1:
            x_1 = [P3, P4, P5]
            for i in range(self.nl):
                x_1[i] = torch.cat((self.cv2_1[i](x_1[i]), self.cv3_1[i](x_1[i])), 1)
            if self.shape != shape:
                self.anchors_1, self.strides_1 = (x_1.transpose(0, 1) for x_1 in make_anchors(x_1, self.stride, 0.5))
                # self.shape = shape
            box_1, cls_1 = torch.cat([xi.view(shape[0], self.no[0], -1) for xi in x_1], 2).split(
                (self.reg_max * 4, self.num_classes[0]), 1)
            dbox_1 = self.dfl_1(box_1)
            # return [[dbox_1, cls_1, x_1, self.anchors_1.to(dbox_1.device), self.strides_1.to(dbox_1.device)]]
        if self.task_number >= 2:
            x_2 = [P3, P4, P5]
            for i in range(self.nl):
                x_2[i] = torch.cat((self.cv2_2[i](x_2[i]), self.cv3_2[i](x_2[i])), 1)
            if self.shape != shape:
                self.anchors_2, self.strides_2 = (x_2.transpose(0, 1) for x_2 in make_anchors(x_2, self.stride, 0.5))
                # self.shape = shape
            box_2, cls_2 = torch.cat([xi.view(shape[0], self.no[1], -1) for xi in x_2], 2).split(
                (self.reg_max * 4, self.num_classes[1]), 1)
            dbox_2 = self.dfl_2(box_2)
        if self.task_number>=3:
            x_3 = [P3, P4, P5]
            for i in range(self.nl):
                x_3[i] = torch.cat((self.cv2_3[i](x_3[i]), self.cv3_3[i](x_3[i])), 1)
            if self.shape != shape:
                self.anchors_3, self.strides_3 = (x_3.transpose(0, 1) for x_3 in make_anchors(x_3, self.stride, 0.5))
                # self.shape = shape
            box_3, cls_3 = torch.cat([xi.view(shape[0], self.no[2], -1) for xi in x_3], 2).split(
                (self.reg_max * 4, self.num_classes[2]), 1)
            dbox_3 = self.dfl_3(box_3)
        if self.task_number>=4:
            x_4 = [P3, P4, P5]
            for i in range(self.nl):
                x_4[i] = torch.cat((self.cv2_4[i](x_4[i]), self.cv3_4[i](x_4[i])), 1)
            if self.shape != shape:
                self.anchors_4, self.strides_4 = (x_4.transpose(0, 1) for x_4 in make_anchors(x_4, self.stride, 0.5))
                # self.shape = shape
            box_4, cls_4 = torch.cat([xi.view(shape[0], self.no[3], -1) for xi in x_4], 2).split(
                (self.reg_max * 4, self.num_classes[3]), 1)
            dbox_4 = self.dfl_4(box_4)
        if self.task_number >= 5:
            x_5 = [P3, P4, P5]
            for i in range(self.nl):
                x_5[i] = torch.cat((self.cv2_5[i](x_5[i]), self.cv3_5[i](x_5[i])), 1)
            if self.shape != shape:
                self.anchors_5, self.strides_5 = (x_5.transpose(0, 1) for x_5 in make_anchors(x_5, self.stride, 0.5))
                # self.shape = shape
            box_5, cls_5 = torch.cat([xi.view(shape[0], self.no[4], -1) for xi in x_5], 2).split(
                (self.reg_max * 4, self.num_classes[4]), 1)
            dbox_5 = self.dfl_5(box_5)

        if self.task_number == 1:
            return [(dbox_1, cls_1, x_1, self.anchors_1.to(dbox_1.device), self.strides_1.to(dbox_1.device))]
        elif self.task_number == 2:
            return [(dbox_1, cls_1, x_1, self.anchors_1.to(dbox_1.device), self.strides_1.to(dbox_1.device)),
                    (dbox_2, cls_2, x_2, self.anchors_2.to(dbox_2.device), self.strides_2.to(dbox_2.device))]
        elif self.task_number == 3:
            return [(dbox_1, cls_1, x_1, self.anchors_1.to(dbox_1.device), self.strides_1.to(dbox_1.device)),
                    (dbox_2, cls_2, x_2, self.anchors_2.to(dbox_2.device), self.strides_2.to(dbox_2.device)),
                    (dbox_3, cls_3, x_3, self.anchors_3.to(dbox_3.device), self.strides_3.to(dbox_3.device))]
        elif self.task_number == 4:
            return [(dbox_1, cls_1, x_1, self.anchors_1.to(dbox_1.device), self.strides_1.to(dbox_1.device)),
                    (dbox_2, cls_2, x_2, self.anchors_2.to(dbox_2.device), self.strides_2.to(dbox_2.device)),
                    (dbox_3, cls_3, x_3, self.anchors_3.to(dbox_3.device), self.strides_3.to(dbox_3.device)),
                    (dbox_4, cls_4, x_4, self.anchors_4.to(dbox_4.device), self.strides_4.to(dbox_4.device))]
        elif self.task_number == 5:
            return [(dbox_1, cls_1, x_1, self.anchors_1.to(dbox_1.device), self.strides_1.to(dbox_1.device)),
                    (dbox_2, cls_2, x_2, self.anchors_2.to(dbox_2.device), self.strides_2.to(dbox_2.device)),
                    (dbox_3, cls_3, x_3, self.anchors_3.to(dbox_3.device), self.strides_3.to(dbox_3.device)),
                    (dbox_4, cls_4, x_4, self.anchors_4.to(dbox_4.device), self.strides_4.to(dbox_4.device)),
                    (dbox_5, cls_5, x_5, self.anchors_5.to(dbox_5.device), self.strides_5.to(dbox_5.device))]

if __name__ == '__main__':
    model =  YoloBody(640, [2,4,6,8,10], 's', pretrained=False)
    model.eval()
    input = torch.rand(2,3,640,640)
    output = model(input)
    print(output[0][0].shape)
    print(output[0][1].shape)
    print(output[1][0].shape)
    print(output[1][1].shape)
    print(output[2][0].shape)
    print(output[2][1].shape)


