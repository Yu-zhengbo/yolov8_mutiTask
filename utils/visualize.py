import wandb
import torch
from PIL import Image,ImageDraw
import numpy as np
from torchvision.ops import nms

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    # -----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    # -----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        # -----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        # -----------------------------------------------------------------#
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

def non_max_suppression(prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5,
                        nms_thres=0.4):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        class_conf, class_pred = torch.max(image_pred[:, 4:4 + num_classes], 1, keepdim=True)
        conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float()), 1)
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4],
                nms_thres
            )
            max_detections = detections_class[keep]
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4] = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return output


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    # 左上右下
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

def decode_box(inputs):
    # dbox  batch_size, 4, 8400
    # cls   batch_size, 20, 8400
    dbox, cls, _, anchors, strides = inputs

    # dbox = torch.from_numpy(dbox)
    # cls = torch.from_numpy(cls)

    # anchors = torch.from_numpy(anchors)
    # strides = torch.from_numpy(strides)

    # 获得中心宽高坐标
    dbox = dist2bbox(dbox, anchors.unsqueeze(0), xywh=True, dim=1) * strides
    y = torch.cat((dbox, cls.sigmoid()), 1).permute(0, 2, 1)
    # 进行归一化，到0~1之间
    y[:, :, :4] = y[:, :, :4] / torch.Tensor([640,640,640,640]).to(y.device)
    return y
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def preprocess_input(image):
    image /= 255.0
    return image

def visual_gt_pred(model_train, class_names, num_classes,visual_list,input_shape = [640, 640],letterbox_image=True,
                   confidence=0.5,nms_iou=0.3):
    model_train = model_train.eval()

    all_names = []
    for i in class_names:
        all_names.extend(i)

    num_add_gt = 0

    for qq,(class_name,class_num,annotation_line) in enumerate(zip(class_names, num_classes,visual_list)):

        line = annotation_line.split()
        image = Image.open(line[0])
        image_gt = image.copy()
        gts = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (input_shape[0], input_shape[0]), letterbox_image)
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        image_data = torch.from_numpy(image_data).cuda()

        with torch.no_grad():
            outputs = model_train(image_data)
            outputs = [decode_box(i) for i in outputs]
            results = [non_max_suppression(outputs[i], num_classes[i], input_shape,
                                           image_shape, letterbox_image, conf_thres=confidence,
                                           nms_thres=nms_iou) for i in range(len(outputs))]

            preds = []
            class_add = 0
            for index_pred,bbox in enumerate(results):
                if bbox[0] is None:
                    class_add += len(class_names[index_pred])
                    continue
                for box in bbox[0].tolist():
                    min_y, min_x, max_y, max_x = box[:4]
                    class_id = int(box[5]) + class_add  # 分类ID
                    box_caption = all_names[class_id]  # f"{class_id}"  # 可以将类名或其他信息用作标题
                    class_score = box[4]  # 分类得分

                    # 每个Boxes对象需要的参数是 xmin, ymin, xmax, ymax, class_id, box_caption, score, domain
                    preds.append({
                        "position": {
                        "minX": min_x,
                        "maxX": max_x,
                        "minY": min_y,
                        "maxY": max_y,
                        },
                        "class_id": class_id,
                        "box_caption": f"{box_caption} {class_score:.2f}",#box_caption,# "Score: %f" % class_score,
                        "domain": "pixel",
                        "scores": {
                            "score": class_score,
                        },
                        })
                class_add += len(class_names[index_pred])
            # Image对象接受image和box数据作为参数，并将其可视化
            wandb_image_pred = wandb.Image(image, boxes={'predictions':{'box_data':preds,'class_labels':dict(zip(range(len(all_names)),all_names))}},mode='L')
            # wandb.log({'wandb_image_pred': wandb_image_pred})







            gtss = []
            for bbox in gts:
                min_x, min_y, max_x, max_y = bbox[:4]
                class_id = int(bbox[4]) + num_add_gt  # 分类ID
                box_caption = all_names[class_id] #f"{class_id}"  # 可以将类名或其他信息用作标题
                class_score = 1.0  # 分类得分

                # 每个Boxes对象需要的参数是 xmin, ymin, xmax, ymax, class_id, box_caption, score, domain
                gtss.append({
                    "position": {
                        "minX": float(min_x),
                        "maxX": float(max_x),
                        "minY": float(min_y),
                        "maxY": float(max_y),
                    },
                    "class_id": class_id,
                    "box_caption": f"{box_caption} {class_score:.2f}",  # "Score: %f" % class_score,
                    "domain": "pixel",
                    "scores": {
                        "score": class_score,
                    },
                })

            # Image对象接受image和box数据作为参数，并将其可视化
            wandb_image_gt = wandb.Image(image_gt, boxes={'predictions':{'box_data':gtss,'class_labels':dict(zip(range(len(all_names)),all_names))}},mode='L')
            # wandb.log({'wandb_image_gt': wandb_image_gt})

            wandb.log({'pred_of_task_{0}'.format(qq+1): wandb_image_pred,'gt_of_task_{0}'.format(qq+1): wandb_image_gt})
            num_add_gt += class_num


