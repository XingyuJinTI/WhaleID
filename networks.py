#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:57:52 2019

@author: xingyu
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.models as models

# EembeddingNet ResNet50
class EmbeddingNet(nn.Module):
    
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        
        self.model = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu, resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4,resnet.avgpool)#,resnet.fc)
        # Fix blocks
        fixed_blocks = 1
        for p in self.model[0].parameters(): p.requires_grad=False
        for p in self.model[1].parameters(): p.requires_grad=False
        if fixed_blocks >= 3:
            for p in self.model[6].parameters(): p.requires_grad=False
        if fixed_blocks >= 2:
            for p in self.model[5].parameters(): p.requires_grad=False
        if fixed_blocks >= 1:
            for p in self.model[4].parameters(): p.requires_grad=False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False

        self.model.apply(set_bn_fix)
        
        resnet.fc = nn.Linear(2048, 5004, bias=True)
        self.last_layer = resnet.fc
        

    def forward(self, x):    
        x = self.model(x)
        x = x.view(-1, 2048)
        return self.last_layer(x)
    
# Siamese Net

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

# ContrastiveLoss

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()
    
