#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:04:40 2019

@author: xingyu
"""
from torch.utils.data import DataLoader,Dataset
import albumentations
from albumentations import torch as AT

import numpy as np
import random
from PIL import Image
import pandas as pd

# Siamese Dataset
class SiameseDataset(Dataset):
    
    def __init__(self,datafolder, df, bbox_df, datatype='train', transform = None):
        self.datafolder = datafolder
        self.df = df
        self.bbox_df = bbox_df
        self.datatype = datatype
        self.transform = transform
        
    def __getitem__(self,idx):
        # not selecting 'new_whale' for anchor image.
        img0_idx = random.choice(self.df[self.df.Id != 'new_whale'].index.values)
        
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            img1_idx = random.choice(self.df[self.df.Id == self.df.Id[img0_idx]].index.values) 
        else:
            img1_idx = random.choice(self.df[self.df.Id != self.df.Id[img0_idx]].index.values)
        
        img0_path = self.df.loc[img0_idx,'Image']
        img1_path = self.df.loc[img1_idx,'Image']
        
        bbox0 = bbox_df.loc[bbox_df.Image==img0_path,:].values[0,1:]
        bbox1 = bbox_df.loc[bbox_df.Image==img1_path,:].values[0,1:]
        img0_pil = Image.open(os.path.join(self.datafolder, img0_path)).crop(bbox0).convert('RGB')
        img1_pil = Image.open(os.path.join(self.datafolder, img1_path)).crop(bbox1).convert('RGB')
        img0 = np.array(img0_pil)
        img1 = np.array(img1_pil)
             
        image0 = self.transform(image=img0)['image']
        image1 = self.transform(image=img1)['image']
        #plt.imshow(image0)
        return image0, image1 , torch.from_numpy(np.array([int(self.df.Id[img0_idx] != self.df.Id[img1_idx])],dtype=np.float32))
    
    def __len__(self):
        return(self.df.shape[0])

