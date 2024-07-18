import torch
from torch.utils.data import Dataset
import os
import cv2
import xml.etree.ElementTree as ET
import yaml

class VOCDataSet(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annots = list(sorted(os.listdir(os.path.join(root, "annotations"))))
        cfg = yaml.load(open("cfg.yaml", "r"), Loader=yaml.FullLoader)
        names = cfg["names"]
        self.label_dict = {}
        for i, name in enumerate(names):
            self.label_dict[name] = i

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annot_path = os.path.join(self.root, "annotations", self.annots[idx])
        img = cv2.imread(img_path)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        boxes = []
        labels = []

        tree = ET.parse(annot_path)
        root = tree.getroot()
        for obj in root.iter('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            x_min = int(bndbox.find('xmin').text)
            y_min = int(bndbox.find('ymin').text)
            x_max = int(bndbox.find('xmax').text)
            y_max = int(bndbox.find('ymax').text)
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.label_dict[name])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        return img, target

    def __len__(self):
        return len(self.imgs)