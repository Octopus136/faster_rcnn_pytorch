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
from utils import nms, ap_per_class

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def launch(train_root, test_root, batch_size, is_cuda, num_epochs, save_path):
    train_data_loader = DataLoader(VOCDataSet(train_root), batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    test_data_loader = DataLoader(VOCDataSet(test_root), batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    
    y = yaml.load(open("cfg.yaml", "r"), Loader=yaml.FullLoader)
    nc = y["nc"]

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
    
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    best_map = 0

    for epoch in range(num_epochs):
        model.train()
        logger.info(('\n' + '%10s' * 4) % ('Epoch', 'gpu_mem', 'labels', 'img_size'))
        pbar = tqdm.tqdm(train_data_loader, total=len(train_data_loader))
        for images, targets in pbar:
            images = list(torch.tensor(image).to(device) for image in images)
            images = [image / 255.0 for image in images]
            images = [image.permute(2, 0, 1) for image in images]
            label_len = len(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' * 2 + '%10.4g' * 2) % (
                    '%g/%g' % (epoch, num_epochs - 1), mem, label_len, images[0].shape[-1])
            pbar.set_description(s)
        
        lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
            seen, nt = 0, 0
            for images, targets in tqdm.tqdm(test_data_loader, total=len(test_data_loader), desc=s):
                images = list(torch.tensor(image).to(device) for image in images)
                images = [image / 255.0 for image in images]
                images = [image.permute(2, 0, 1) for image in images]
                gt = []
                for t in targets:
                    boxes = t["boxes"]
                    labels = t["labels"]
                    gt.append(labels, boxes[0], boxes[1], boxes[2], boxes[3]) # labels, x1, y1, x2, y2
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                out = model(images, targets)
                batch_map = 0
                for o in out:
                    boxes = []
                    seen += 1
                    for box, label, score in zip(o["boxes"], o["labels"], o["scores"]):
                        x1, y1, x2, y2 = box
                        conf = score
                        cls = label
                        boxes.append([cls, x1, y1, x2, y2, conf])

                    boxes = nms(boxes, 0.6)
                    nt += len(boxes)
                    p, r, ap50, ap = ap_per_class(boxes, gt, nc)
                    batch_map += ap
                    mp += p
                    mr += r
                    map50 += ap50
                    map += ap
                batch_map /= len(out)
                if batch_map > best_map:
                    best_map = batch_map
                    torch.save(model.state_dict(), f'best_{epoch}.pt')
            
            mp, mr, map50, map = mp / seen, mr / seen, map50 / seen, map / seen
            pf = '%20s' + '%12i' * 2 + '%12.3g' * 4
            print(pf % ('all', seen, nt, mp, mr, map50, map))

    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True, help="Path to the root directory of the training data.")
    parser.add_argument("--test_root", type=str, required=True, help="Path to the root directory of the testing data.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--is_cuda", type=bool, default=True, help="Whether to use CUDA.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the model.")
    args = parser.parse_args()
    launch(args.train_root, args.test_root, args.batch_size, args.is_cuda, args.num_epochs, args.save_path)

