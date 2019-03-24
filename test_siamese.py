#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:18:41 2019

@author: xingyu
"""

import os

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

train_full = pd.read_csv("train.csv")
test_df = pd.read_csv("sample_submission.csv")
id_counts = train_full.Id.value_counts()
bbox_df = pd.read_csv("bounding_boxes.csv")

RESIZE_H = 224
RESIZE_W = 224


data_transforms_test = albumentations.Compose([
    albumentations.Resize(RESIZE_H, RESIZE_W),
    albumentations.Normalize(),
    AT.ToTensor()
])

# load the best model and initialize EmbeddingNet
siamese_net = torch.load("best_net.pth")
state_dict = siamese_net.state_dict()
embed_net = EmbeddingNet()
temp_dict = {}
for key in state_dict.keys():
    if key.startswith("embedding_net"):
        temp_dict[key[14:len(key)]] = state_dict[key]
embed_net.load_state_dict(temp_dict)


# function to generate embedding
def getEmbedding(file_path,x):
    file_name = os.path.join(file_path,x)
    bbox = bbox_df.loc[bbox_df.Image==x,:].values[0,1:]
    img_pil = Image.open(file_name).crop(bbox).convert('RGB')        
    img = np.array(img_pil)
    image = data_transforms_test(image=img)['image'].unsqueeze(0)
    vector = embed_net(image) 
    return vector

# test
embed_net.eval()
train_embed_dataset = train_full.assign(embedding = train_full['Image'].apply(lambda x : getEmbedding('train/',x)))
test_embed_dataset = test_df.assign(embedding = test_df['Image'].apply(lambda x : getEmbedding('test/',x)))
pickle.dump(train_embed_dataset,open( "train_full_embed.p",'wb'))
pickle.dump(test_embed_dataset,open( "test_df_embed.p",'wb'))


