import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import os
from torch.utils.data import DataLoader
from vocdata import VOCDataSet
from model import FasterRCNN
import yaml
import argparse
import tqdm
import logging
import cv2
from utils import nms, color_list, plot_one_box

colors = color_list()

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def launch(weight, source, is_cuda, box_thresh, nms_thresh, save_path, thickness):
    images = os.listdir(source)
    
    y = yaml.load(open("cfg.yaml", "r"), Loader=yaml.FullLoader)
    nc = y["nc"]
    names = y["names"]

    # model = FasterRCNN(num_classes=nc, backbone_name='resnet')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, nc)

    if is_cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("CUDA is not available.")
            return
    else:
        device = torch.device("cpu")
    
    model.load_state_dict(torch.load(weight))
    model.to(device)

    model.eval()
    with torch.no_grad():
        for image in tqdm.tqdm(images):
            nimage = cv2.imread(os.path.join(source, image))
            timage = torch.tensor(nimage).to(device)
            timage = timage / 255.0
            timage = timage.permute(2, 0, 1)
            timage = timage.reshape(1, *timage.shape)
            result = model(timage)
            boxes = []
            for box, label, score in zip(result[0]["boxes"], result[0]["labels"], result[0]["scores"]):
                if score > box_thresh:
                    x1, y1, x2, y2 = box
                    conf = score
                    cls = label
                    boxes.append([cls, x1, y1, x2, y2, conf])
            nms(boxes, nms_thresh)

            for box in boxes:
                cls, x1, y1, x2, y2, conf = box
                color = colors[cls]
                label = names[cls]
                plot_one_box([x1, y1, x2, y2], nimage, color, label, line_thickness=thickness)
            
            cv2.imwrite(os.path.join(save_path, image), nimage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, required=True, help="Path to the weight.")
    parser.add_argument("--source", type=str, required=True, help="Path to the source.")
    parser.add_argument("--is_cuda", type=bool, default=True, help="Whether to use CUDA.")
    parser.add_argument("--box_thresh", type=float, default=0.45, help="Box threshold.")
    parser.add_argument("--nms_thresh", type=float, default=0.45, help="Label threshold.")
    parser.add_argument("--save_path", type=str, default="output", help="Path to save the output.")
    parser.add_argument("--thickness", type=int, default=3, help="Line thickness.")
    args = parser.parse_args()
    launch(args.weight, args.source, args.is_cuda, args.box_thresh, args.nms_thresh, args.save_path, args.thickness)

