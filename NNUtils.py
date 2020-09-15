import numpy as np
import torch
import math
import pydicom
from PIL import Image
from tqdm import tqdm_notebook

def iou(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    w1 = x12 - x11
    h1 = y12 - y11
    w2 = x22 - x21
    h2 = y22 - y21

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2 - xi1) * (yi2 - yi1)
        union = area1 + area2 - intersect
        return intersect / union


def map_iou(boxes_true, boxes_pred, scores, thresholds=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold

    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image.
                    bbox format: (x1, y1, x2, y2)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image.
                    bbox format: (x1, y1, x2, y2)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output:
        map: mean average precision of the image
    """

    # According to the introduction, images with no ground truth bboxes will not be
    # included in the map score unless there is a false positive detection (?)

    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None

    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]

    map_total = 0

    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1  # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1  # bt has no match, count as FN

        fp = len(boxes_pred) - len(matched_bt)  # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m

    return map_total / len(thresholds)


def evaluate(model, data_loader, device):
    boxes_pred = []
    boxes_true = []
    scores = []
    summ = 0
    count = 0
    with torch.no_grad():
        model.eval()

        for images, targets in tqdm_notebook(data_loader):
            boxes_true_mini_batch = [np.array(item["boxes"]) for item in targets]
            labels_true_mini_batch = [np.array(item["labels"]) for item in targets]
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            torch.cuda.synchronize()
            outputs = model(images)
            outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            boxes_pred_mini_batch = [np.array(res["boxes"].to("cpu")) for res in outputs]
            scores_mini_batch = [np.array(res["scores"].to("cpu")) for res in outputs]
            labels_mini_batch = [np.array(res["labels"].to("cpu")) for res in outputs]

            for img_num in range(len(images)):
                # Если на картинке нет пневмонии
                if np.all(labels_true_mini_batch[img_num] == 0):
                    if (labels_mini_batch[img_num].size == 0) or np.all(labels_mini_batch[img_num] == 0):
                        continue
                    else:
                        # Мы сказали, что есть
                        count += 1
                else:
                    # Если пневмония есть считаем map_iou
                    curr_map_iou = map_iou(boxes_true_mini_batch[img_num],
                                           boxes_pred_mini_batch[img_num],
                                           scores_mini_batch[img_num])
                    summ += curr_map_iou
                    count += 1

    return summ / count


def collate_fn(batch):
    return tuple(zip(*batch))


def cleanup(s_list):
    s_list.pop(0)
    s_list[-1] = s_list[-1][:-1]
    for i in range(len(s_list)-1, -1,-1):
        if s_list[i] in ['\n', ' ', '\t']:
            s_list.pop(i)

def wh_box_to_coord(box):
    x_min = box[0]
    y_min = box[1]
    width = box[2]
    height = box[3]
    return [x_min, y_min, x_min+width, y_min+height]


def parse_submissions(f_names):
    preds = {}
    for name in f_names:
        with open(name) as f:
            f.readline()
            for line in f:
                segments = line.split(',')
                if not segments[0] in preds:
                    preds[segments[0]] = [[], []]
                s = segments[1].split(' ')
                if len(s) == 1:
                    continue
                cleanup(s)
                n_boxes = int(len(s) / 5)
                for i in range(n_boxes):
                    idx = i * 5
                    score = float(s[idx])
                    box = []
                    for j in range(idx + 1, idx + 5):
                        box.append(int(s[j]))
                    preds[segments[0]][0].append(score)
                    preds[segments[0]][1].append(wh_box_to_coord(box))
    return preds

def soft_nms(f_names, sigma=0.5):
    imgs_info = []
    preds = parse_submissions(f_names)
    for item in preds.items():
        key = item[0]
        scores = np.array(item[1][0])
        boxes = item[1][1]
        new_info = {}
        new_info["patient_id"] = key
        new_scores = []
        new_boxes = []
        b_count = len(boxes)
        while b_count > 0:
            m = np.argmax(scores)
            m_b = boxes[m]
            m_val = scores[m]
            scores[m] = -1
            boxes[m] = []
            b_count -= 1
            new_scores.append(m_val)
            new_boxes.append(m_b)
            for i, s in enumerate(scores):
                if s > -1:
                    iou_score = iou(boxes[i], m_b)
                    scores[i] = s * math.exp(-(iou_score ** 2)/sigma)
        new_info["boxes"] = np.array(new_boxes)
        new_info["scores"] = new_scores
        imgs_info.append(new_info)
    return imgs_info

def load_test_image(img_path, img_size):
    ds = pydicom.read_file(img_path)
    image = ds.pixel_array
    image = np.array(Image.fromarray(image).resize((img_size, img_size)))
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1)
    image = np.rollaxis(image, 2, 0) / 255
    return torch.Tensor(image)


def get_test_predictions(model, test_images, device, img_size):
    """
    Предсказания для теста
    """
    sub = []
    min_conf = 0
    imgs_info = []
    scale_factor = 1024 / img_size
    with torch.no_grad():
        model.eval()

        for img_path in tqdm_notebook(test_images):
            images = [load_test_image(img_path, img_size)]
            images = list(img.to(device) for img in images)

            torch.cuda.synchronize()
            outputs = model(images)
            outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

            boxes_pred_mini_batch = [np.array(res["boxes"].to("cpu")) for res in outputs]
            scores_mini_batch = [np.array(res["scores"].to("cpu")) for res in outputs]

            for i in range(len(images)):
                patient_id = img_path.split(".dcm")[0].split("/")[-1]
                img_info = dict()
                img_info["patient_id"] = patient_id
                img_info["boxes"] = boxes_pred_mini_batch[i]
                img_info["scores"] = scores_mini_batch[i]
                imgs_info.append(img_info)
    return imgs_info


def get_sub_list(imgs_info, img_size, min_conf=0.7):
    """
    Записываем предсказания в правильном формате
    """
    sub = []
    scale_factor = 1024 / img_size
    for img_info in imgs_info:
        patient_id = img_info["patient_id"]
        boxes_pred_mini_batch = img_info["boxes"]
        scores_mini_batch = img_info["scores"]

        result_str = "{},".format(patient_id)
        for bbox_num in range(boxes_pred_mini_batch.shape[0]):
            if scores_mini_batch[bbox_num] > min_conf:
                result_str += " {:1.2f} ".format(np.round(scores_mini_batch[bbox_num], 2))
                x_min = int(np.round(boxes_pred_mini_batch[bbox_num, 0] * scale_factor))
                y_min = int(np.round(boxes_pred_mini_batch[bbox_num, 1] * scale_factor))
                width = int(np.round(boxes_pred_mini_batch[bbox_num, 2] * scale_factor)) - x_min
                height = int(np.round(boxes_pred_mini_batch[bbox_num, 3] * scale_factor)) - y_min
                result_str += "{} {} {} {}".format(x_min, y_min, width, height)
        sub.append(result_str + "\n")
    return sub


def write_submission(sub, filename="submission.csv"):
    with open(filename, mode="w") as f:
        header = "patientId,PredictionString\n"
        f.write(header)
        for line in sub:
            f.write(line)


