import onnxruntime as rt
import numpy as np
from PIL import Image,ImageDraw, ImageFont
import time
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


def bbox_iou(box1, box2, xywh=True, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    #
    # (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.reshape(4, -1), box2.reshape(4, -1)
    # w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    # b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    # b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    #
    # inter = np.clip((np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)), a_min=0, a_max=None) * np.clip(
    #     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)), a_min=0, a_max=None)
    #
    # union = w1 * h1 + w2 * h2 - inter + eps
    # iou = inter / union

    box1 = [box1[0]-box1[2]/2, box1[1]-box1[3]/2, box1[0]+box1[2]/2, box1[1]+box1[3]/2]
    box2[:,0] = box2[:,0]-box2[:,2]/2
    box2[:,1] = box2[:,1]-box2[:,3]/2
    box2[:,2] = box2[:,0]+box2[:,2]
    box2[:,3] = box2[:,1]+box2[:,3]

    inter_x1 = np.maximum(box1[ 0], box2[:, 0])
    inter_y1 = np.maximum(box1[1], box2[:, 1])
    inter_x2 = np.minimum(box1[2], box2[:, 2])
    inter_y2 = np.minimum(box1[3], box2[:, 3])

    # Calculate the area of intersect between box1 and box2
    inter_area = np.maximum(inter_y2 - inter_y1, 0) * np.maximum(inter_x2 - inter_x1, 0)

    # Calculate the area of union between box1 and box2
    box1_area = (box1[3] - box1[ 1]) * (box1[2] - box1[0])
    box2_area = (box2[:, 3] - box2[:, 1]) * (box2[:, 2] - box2[:, 0])
    union_area = box1_area + box2_area - inter_area + 1e-16

    # compute the IoU
    iou = inter_area / union_area

    return iou  # IoU

def non_max_suppression(prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5,
                        nms_thres=0.4):
    # box_corner = prediction.new(prediction.shape)
    box_corner = np.zeros_like(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        class_conf, class_pred = np.max(image_pred[:, 4:4 + num_classes], 1, keepdims=True), \
            np.argmax(image_pred[:, 4:4 + num_classes], 1, keepdims=True)
        conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.shape[0]:
            continue
        detections = np.concatenate((image_pred[:, :4], class_conf, class_pred), 1)
        unique_labels = np.unique(detections[:, -1])

        # if prediction.is_cuda:
        #     unique_labels = unique_labels.cuda()
        #     detections = detections.cuda()

        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            # keep = nms(
            #     torch.from_numpy(detections_class[:, :4]),
            #     torch.from_numpy(detections_class[:, 4]),
            #     nms_thres
            # ).numpy()
            # max_detections = detections_class[keep]

            conf_sort_index = np.argsort(-1*detections_class[:, 4]*detections_class[:, 5])
            detections_class = detections_class[conf_sort_index]
            # 进行非极大抑制
            max_detections = []
            while detections_class.shape[0]:
                # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                max_detections.append(np.expand_dims(detections_class[0],0))
                if len(detections_class) == 1:
                    break
                ious = bbox_iou(detections_class[0,:4].copy(), detections_class[1:,:4].copy())
                detections_class = detections_class[1:][ious < nms_thres]
            # 堆叠
            max_detections = np.concatenate(max_detections)

            output[i] = max_detections if output[i] is None else np.concatenate((output[i], max_detections))

        if output[i] is not None:
            output[i] = output[i]
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
    input_name = 'images'
    # ===========================================



    # t1 = time.time()
    # for _ in range(100):
    outputs = sess.run(None, {input_name: image_data})

    results = [non_max_suppression(outputs[i], num_classes[i], input_shape,
                                   image_shape, letterbox_image, conf_thres=confidence,
                                   nms_thres=nms_iou) for i in range(len(outputs))]
    # t2 = time.time()
    # print(100/(t2-t1))

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


