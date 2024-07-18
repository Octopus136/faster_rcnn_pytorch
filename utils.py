import cv2
import matplotlib
import random
import numpy as np

def calculate_iou(box1, box2):
    # box1: [x1, y1, x2, y2]
    # box2: [x1, y1, x2, y2]
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    x5 = max(x1, x3)
    y5 = max(y1, y3)
    x6 = min(x2, x4)
    y6 = min(y2, y4)
    if x5 >= x6 or y5 >= y6:
        return 0
    area_inter = (x6 - x5) * (y6 - y5)
    area_union = area1 + area2 - area_inter
    return area_inter / area_union

def nms(boxes, thres):
    # boxes: [[cls, x1, y1, x2, y2, conf], ...]
    boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if i >= len(boxes) or j >= len(boxes):
                break
            iou = calculate_iou(boxes[i][1:5], boxes[j][1:5])
            if iou > thres:
                boxes.pop(j)
                j -= 1
    return boxes

def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def precision_recall(gt, pred, iou_threshold):
    # gt: [[cls, x1, y1, x2, y2], ...]
    # pred: [[cls, x1, y1, x2, y2, conf], ...]
    gt = np.array(gt)
    pred = np.array(pred)

    tp = 0  # true positive
    fp = 0  # false positive
    fn = 0  # false negative

    for pred_box in pred:
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt):
            if gt_box[0] == pred_box[0]:  # check if the class is the same
                iou_value = calculate_iou(gt_box[1:], pred_box[1:5])
                if iou_value > best_iou:
                    best_iou = iou_value
                    best_gt_idx = gt_idx

        if best_iou > iou_threshold:
            tp += 1
            gt = np.delete(gt, best_gt_idx, 0)
        else:
            fp += 1

    fn = len(gt)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    return precision, recall

def calculate_ap(precisions, recalls):
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    recalls = np.concatenate(([0.0], recalls, [1.0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return ap

def mean_average_precision(gt, pred, iou_thresholds, num_classes):
    aps = []
    for iou_threshold in iou_thresholds:
        precision_list = []
        recall_list = []

        for cls in range(num_classes):
            gt_cls = [box for box in gt if box[0] == cls]
            pred_cls = [box for box in pred if box[0] == cls]
            pred_cls = sorted(pred_cls, key=lambda x: x[5], reverse=True)  # sort by confidence

            if not gt_cls:
                continue

            precisions = []
            recalls = []
            tp = 0
            fp = 0

            for pred_box in pred_cls:
                best_iou = 0
                best_gt_idx = -1
                for gt_idx, gt_box in enumerate(gt_cls):
                    iou_value = calculate_iou(gt_box[1:], pred_box[1:5])
                    if iou_value > best_iou:
                        best_iou = iou_value
                        best_gt_idx = gt_idx

                if best_iou > iou_threshold:
                    tp += 1
                    gt_cls.pop(best_gt_idx)
                else:
                    fp += 1

                fn = len(gt_cls)
                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0

                precisions.append(precision)
                recalls.append(recall)

            precision_list.append(np.array(precisions))
            recall_list.append(np.array(recalls))

        # Calculate AP for this IoU threshold
        all_precisions = np.concatenate(precision_list) if precision_list else np.array([0])
        all_recalls = np.concatenate(recall_list) if recall_list else np.array([0])
        ap = calculate_ap(all_precisions, all_recalls)
        aps.append(ap)

    return np.mean(aps)

def ap_per_class(boxes, gt, nc):
    precision, recall = precision_recall(gt, boxes, 0.5)
    map50 = mean_average_precision(gt, boxes, [0.5], nc)
    map50_95 = mean_average_precision(gt, boxes, np.linspace(0.5, 0.95, 10), nc)
    return precision, recall, map50, map50_95

