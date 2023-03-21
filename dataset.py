import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
from pprint import pprint
from time import perf_counter as pf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config import anchors
import albumentations as A
import albumentations.pytorch
import numpy as np


classes = ['horse', 'person', 'bottle', 'dog', 'tvmonitor', 'car', 'aeroplane', 'bicycle', 'boat', 'chair', 'diningtable', 'pottedplant', 'train', 'cat', 'sofa', 'bird', 'sheep', 'motorbike', 'bus', 'cow']
def get_bboxlist(bnds):
    bboxes = []
    for b in bnds:
        xmin = int(b['bndbox']['xmin'])
        ymin = int(b['bndbox']['ymin'])
        xmax = int(b['bndbox']['xmax'])
        ymax = int(b['bndbox']['ymax'])
        bbox_class = classes.index(b['name'])
        bboxes.append([xmin, ymin, xmax, ymax, bbox_class])
    return bboxes

def same_xy_iou(box, anchors):
    intersections = torch.where(box < anchors, box, anchors).prod(dim=1)
    unions = box.prod() + anchors.prod(dim=1)
    return intersections / (unions - intersections)


class YoloDataset(Dataset):
    def __init__(self, data, S=[13, 26, 52], C=20, image_size=416, min_iou=0.35):
        super().__init__()
        self.S = S
        self.C = C
        self.data = data
        self.image_size = image_size
        self.min_iou = min_iou

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # ASSUMES 3 SCALED PREDICTIONS AND 9 ANCHORS TOTAL WITH 3 PER SCALED PREDICTION
        image, o = self.data[idx]
        image = np.array(image)

        bnds = o['annotation']['object']
        bboxes = get_bboxlist(bnds)
        transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.RandomCrop(self.image_size, self.image_size),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.07),
            A.HorizontalFlip(),
            A.pytorch.transforms.ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc'))

        augmentations = transform(image=image, bboxes=bboxes)
        image = augmentations["image"]
        bboxes = augmentations["bboxes"]
        image = image / 255.0
        
        for i, box in enumerate(bboxes):
            # xyxy format to xywh
            width = box[2] - box[0]
            height = box[3] - box[1]
            xcenter = box[0] + width / 2
            ycenter = box[1] + height / 2
            bboxes[i] = [xcenter, ycenter, width, height, box[4]]
            
        bboxes = torch.tensor(bboxes)
        bboxes[:, :4] /= self.image_size

        labels = [torch.zeros(S, S, 3, 6) for S in self.S]
        for box in bboxes:
            x, y, w, h = box[:4]
            ious, iou_indices = same_xy_iou(box[:2], anchors).sort(descending=True)
            good_indices = iou_indices[ious > self.min_iou].tolist()

            if not good_indices: # If no anchors have good enough iou just set best one as target
                good_indices.append(iou_indices[0])

            for indice in good_indices:
                scale_idx = indice // 3
                anchor_idx = indice % 3
                S = self.S[scale_idx]

                i, j = int(x * S), int(y * S)
                xcell, ycell = x * S - i, y * S - j

                current_box = torch.tensor([xcell, ycell, w, h, 1, box[4]])
                labels[scale_idx][i, j, anchor_idx] = current_box


        return image, labels

def get_dataset(batch_size=32, S=[13, 26, 52], C=20, image_size=416):
    data = torchvision.datasets.VOCDetection("data", '2012', 'train', download=False)
    dataloader = DataLoader(YoloDataset(data, S, C, image_size), batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
    return dataloader

