
import os
import csv

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader,Dataset
import albumentations
from albumentations import torch as AT

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import random
from PIL import Image

from networks import EmbeddingNet, SiameseNet, ContrastiveLoss
from datasets import SiameseDataset




# ## Data Path and Data Transforms

train_full = pd.read_csv("train.csv")
test_df = pd.read_csv("sample_submission.csv")

id_counts = train_full.Id.value_counts()

valid_df = train_full.loc[train_full.Id.isin(id_counts[id_counts>5].index.values),:].sample(frac=0.1)

train_df = train_full.loc[~train_full.index.isin(valid_df.index.values),:]

test_df = pd.read_csv("sample_submission.csv")

bbox_df = pd.read_csv("bounding_boxes.csv")

RESIZE_H = 224
RESIZE_W = 224

data_transforms = albumentations.Compose([
    albumentations.Resize(RESIZE_H, RESIZE_W),
    albumentations.HorizontalFlip(),
    albumentations.OneOf([
        albumentations.RandomContrast(),
        albumentations.RandomBrightness(),
        albumentations.Blur()
    ]),
    albumentations.ShiftScaleRotate(rotate_limit=10, scale_limit=0.15),
    albumentations.JpegCompression(80),
    albumentations.HueSaturationValue(),
    albumentations.Normalize(),
    AT.ToTensor()
])

data_transforms_test = albumentations.Compose([
    albumentations.Resize(RESIZE_H, RESIZE_W),
    albumentations.Normalize(),
    AT.ToTensor()
])


# ## Set up

train_dataset = SiameseDataset(datafolder="train/", 
                                 df=train_df, bbox_df=bbox_df, datatype='train', transform = data_transforms)
valid_dataset = SiameseDataset(datafolder="train/", 
                                 df=valid_df, bbox_df=bbox_df, datatype='train', transform = data_transforms)

train_dataloader = DataLoader(train_dataset,
                        shuffle=True,
                        num_workers=0,
                        batch_size=64)

valid_dataloader = DataLoader(train_dataset,
                        shuffle=False,
                        num_workers=0,
                        batch_size=32)

embed = EmbeddingNet().cuda()
net = SiameseNet(embed).cuda()
#embed = EmbeddingNet()
#net = SiameseNet(embed)
criterion = ContrastiveLoss(margin=0.2)
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

counter = []
loss_history = [] 
iteration_number= 0



epoch_num = 50
loss = 0
tol_loss = [10]
path_best = 'best_net.pth'
path_cur = 'cur_net.pth'
total_num_val = len(valid_df)

for epoch in range(0,epoch_num):
    # train
    net.train()
    for i, data in enumerate(train_dataloader,0):
        img0, img1 , label = data
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        optimizer.zero_grad()
        output1,output2 = net(img0,img1)
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()
        if i %100 == 0 :
            print("Epoch number {} \t Iteration number {} \t Current loss {}\n".format(epoch,iteration_number,loss_contrastive.item()))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())


    # validate
    net.eval()
    with torch.no_grad():
        loss = 0
        for i, data in enumerate(valid_dataloader):
            val_img0, val_img1, val_label = data
            val_img0, val_img1, val_label = val_img0.cuda(), val_img1.cuda(), val_label.cuda()
            val_output1,val_output2 = net(val_img0, val_img1)
            val_loss = criterion(val_output1,val_output2,val_label)
            loss += val_loss.item()
        
        avg_loss = loss/total_num_val
        row = ["{}".format(epoch), "{}".format(avg_loss)]
        with open('validate_loss.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        print("Epoch number {} \t Val loss {}\n".format(epoch, avg_loss))
        if avg_loss < min(tol_loss):
            torch.save(net, path_best)
            print("Checkpoint saved to {}".format(path_best))
        torch.save(net, path_cur)
        print("Checkpoint saved to {}".format(path_cur))
        tol_loss.append(avg_loss)

