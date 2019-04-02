
# coding: utf-8

# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')

import os

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch import optim
from torch.autograd import Variable

import torchvision
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader,Dataset
import albumentations
from albumentations import torch as AT

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import random
import cv2
from PIL import Image
import PIL.ImageOps    
import pickle
from sklearn.metrics.pairwise import euclidean_distances


# In[23]:


p1 = open("train_full_embed.p",'rb')
pp1 = pickle.load(p1)
train_e = pp1.loc[:,'Embedding'].apply(lambda x : (x.squeeze().cpu().numpy()))
#train_e.shape


# In[25]:


p2 = open("test_full_embed.p",'rb')
pp2 = pickle.load(p2)
test_e = pp2.loc[:,'Embedding'].apply(lambda x : (x.squeeze().cpu().numpy()))
#test_e.shape


# In[26]:


train_full = pd.read_csv("train.csv")
test_df = pd.read_csv("sample_submission.csv")


# In[28]:


distance_mat = pd.DataFrame(euclidean_distances(np.stack(train_e.values), np.stack(test_e.values)), columns = pp2.Image.values,index=pp1.Image.values)

Id_Df = pp1[['Image','Id']].set_index('Image')

#def getTopFiveIdValid(x):
#    sortedIds = Id_Df.loc[distance_mat.loc[:,x].sort_values().index.values,'Id'].values
#    topFiveIds = sortedIds[np.sort(np.unique(sortedIds, return_index=True)[1])[:5]]
#    print('yes')
#    return(topFiveIds)
def getTopFiveIdTest(x):
    sortedIds = Id_Df.loc[distance_mat.loc[:,x].sort_values().index.values,'Id'].values
    topFiveIds = ' '.join(sortedIds[np.sort(np.unique(sortedIds, return_index=True)[1])[:5]])
    print('yes')
    return(topFiveIds)

test_df = test_df.assign(Id = pp2.loc[:,'Image'].apply(getTopFiveIdTest))





#test_df =  test_df.assign(Id = pp2.loc[:,'Image'].apply(getTopFiveIdValid))


# In[15]:


test_df[['Image','Id']].to_csv('submission2.csv',index=False)

print('submission2.csv is ok')
