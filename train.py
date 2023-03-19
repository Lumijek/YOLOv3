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

import warnings
warnings.filterwarnings("ignore")
torch.set_printoptions(sci_mode=False)

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")

model = YOLOv3()
criterion = Yolov3Loss().to(device)
epochs = 160
batch_size = 4
classes = ['horse', 'person', 'bottle', 'dog', 'tvmonitor', 'car', 'aeroplane', 'bicycle', 'boat', 'chair', 'diningtable', 'pottedplant', 'train', 'cat', 'sofa', 'bird', 'sheep', 'motorbike', 'bus', 'cow']
dataloader = get_dataset(batch_size)

lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)

def train_network(model, optimizer, criterion, epochs, dataloader, device):
    model = nn.DataParallel(model)
    model = model.to(device)
    cycle = 0
    for epoch in range(epochs):
        if epoch == 10:
            optimizer.param_groups[0]["lr"] = 8e-3
        if epoch == 60:
            optimizer.param_groups[0]["lr"] = 1e-4
        if epoch == 90:
            optimizer.param_groups[0]["lr"] = 1e-5

        for image, labels in tqdm(dataloader):

            image = image.to(device)
            labels = [l.to(device) for l in labels]

            optimizer.zero_grad()

            x1, x2, x3 = model(image)
            print(x1.shape)
            loss_s1, a1, b1, c1 = criterion(x1, labels[0], 0)
            loss_s2, a2, b2, c2 = criterion(x2, labels[1], 1)
            loss_s3, a3, b3, c3 = criterion(x3, labels[2], 2)
            print(labels[0].shape)

            loss = loss_s1 + loss_s2 + loss_s3
            a = a1 + a2 + a3
            b = b1 + b2 + b3
            c = c1 + c2 + c3
            loss.backward()
            optimizer.step()
            if cycle % 10 == 0:
                print("Loss:", loss.item())
                print(a1.item(), b1.item(), c1.item())
                print(a2.item(), b2.item(), c2.item())
                print(a3.item(), b3.item(), c3.item())

            if cycle % 200 == 0:
                torch.save(model.state_dict(), "model.pt")
            cycle += 1

if __name__ == '__main__':
    train_network(model, optimizer, criterion, epochs, dataloader, device)
