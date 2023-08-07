import time

import onnxruntime as rt
import numpy as np
from PIL import Image,ImageDraw, ImageFont
import torch
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

if __name__ == '__main__':
    # 加载模型
    sess = rt.InferenceSession("./model_data/models_withdecode_box.onnx")
    image = Image.open('./3.png')
    image_shape = np.array(image).shape[:2]

    image = cvtColor(image)
    # ---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    # ---------------------------------------------------------#
    image_data = resize_image(image, (640, 640), True)

    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)


    # ===========================================
    num_classes = [4,2]
    input_shape = [640, 640]
    letterbox_image = True
    confidence = 0.5
    nms_iou = 0.3
    class_names = [['speedlimit','crosswalk','trafficlight','stop'],['sagittaria','sagittaria_flower']]

    # ===========================================

    input_name = 'images'


    t1 = time.time()
    for _ in range(100):
        outputs = sess.run(None, {input_name: image_data})
        for i in range(len(outputs)):
            outputs[i] = torch.from_numpy(outputs[i])
        results = [non_max_suppression(outputs[i], num_classes[i], input_shape,
                                       image_shape, letterbox_image, conf_thres=confidence,
                                       nms_thres=nms_iou) for i in range(len(outputs))]
    t2 = time.time()
    print('time: ', 100/(t2 - t1))


    not_None = []
    for i in results:
        if i[0] is not None:
            not_None.append(True)
        else:
            not_None.append(False)
    not_None = np.array(not_None, dtype=bool)
    if  not_None.any():
        top_labels = []
        top_confs = []
        top_boxess = []
        # ---------------------------------------------------------#
        #   将图像进行反绘制
        for i in range(len(not_None)):
            if not_None[i]:
                top_labels.append(np.array(results[i][0][:, 5], dtype='int32'))
                top_confs.append(results[i][0][:, 4])
                top_boxess.append(results[i][0][:, :4])
            else:
                top_labels.append(np.array([], dtype='int32'))
                top_confs.append(np.array([], dtype='float32'))
                top_boxess.append(np.array([], dtype='float32'))

        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))
        # ---------------------------------------------------------#
        for q in range(len(outputs)):
            if not_None[q] == False:
                continue
            for i, c in list(enumerate(top_labels[q])):
                predicted_class = class_names[q][int(c)]
                box = top_boxess[q][i]
                score = top_confs[q][i]

                top, left, bottom, right = box

                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                print(label, top, left, bottom, right)

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(0,128,0))
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(0,128,0))
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                del draw
    image.show()


