import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from  matplotlib import patches
import numpy as np
from pprint import pprint
from tqdm import tqdm
from time import perf_counter as timer

from dataset import *
from loss import Yolov3Loss
from model import YOLOv3
from config import anchors

import warnings

import numpy as np


warnings.filterwarnings("ignore")
torch.set_printoptions(sci_mode=False)

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
model = YOLOv3()
#model = nn.DataParallel(model)
#cp = torch.load("data/model.pt", map_location=torch.device("cpu"))
#model.load_state_dict(cp)
criterion = Yolov3Loss().to(device)
epochs = 50
batch_size = 8
classes = ['horse', 'person', 'bottle', 'dog', 'tvmonitor', 'car', 'aeroplane', 'bicycle', 'boat', 'chair', 'diningtable', 'pottedplant', 'train', 'cat', 'sofa', 'bird', 'sheep', 'motorbike', 'bus', 'cow']
dataloader = get_dataset(batch_size)

lr = 1e-4

optimizer = torch.optim.SGD(model.parameters(), lr, 0.90, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(dataloader), 1e-6)

def train_network(model, optimizer, scheduler, criterion, epochs, dataloader, device):
    model = nn.DataParallel(model)
    model = model.to(device)
    cycle = 0
    for epoch in range(epochs):
        print(epoch)
        i = 0
        for image, labels in tqdm(dataloader):
            print(optimizer.param_groups[0]["lr"])

            image = image.to(device)
            labels = [l.to(device) for l in labels]

            optimizer.zero_grad()
            x1, x2, x3 = model(image)

            loss_s1, a1, b1, c1, d1, e1 = criterion(x1, labels[0], 0)
            loss_s2, a2, b2, c2, d2, e2 = criterion(x2, labels[1], 1)
            loss_s3, a3, b3, c3, d3, e3 = criterion(x3, labels[2], 2)

            loss = loss_s1 + loss_s2 + loss_s3
            loss.backward()

            optimizer.step()
            scheduler.step()

            if cycle % 1 == 0:
                print("Loss:", loss.item())
                print(a1.item(), b1.item(), c1.item(), d1.item(), e1.item())
                print(a2.item(), b2.item(), c2.item(), d2.item(), e2.item())
                print(a3.item(), b3.item(), c3.item(), d3.item(), e3.item())
                print((a1 + a2 + a3).item() / 3, (b1 + b2 + b3).item() / 3, (c1 + c2 + c3).item() / 3)


            if cycle % 200 == 0:
                torch.save(model.state_dict(), "model.pt")
            cycle += 1
            i += 1

if __name__ == '__main__':
    train_network(model, optimizer, scheduler, criterion, epochs, dataloader, device)



