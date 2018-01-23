from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.utils.data as data

import numpy as np
import pandas as pd

class DSB2018Dataset(data.Dataset):

    def __init__(self, id_file, img_file, mask_file, transform=None):
        id_f = open(id_file, 'r')
        self.ids = id_f.readlines()
        self.imgs = np.load(img_file)
        self.masks = np.load(mask_file)
        self.transform = transform

    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx].reshape([256,256])
        id = self.ids[idx]
        
        if self.transform:
            img, mask = self.transform(img, mask)
            
        img = np.transpose(img, (2, 0 ,1)) # transpost to channel first
        
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask.astype(np.int)).long()
        
        return id, img, mask

    def __len__(self):
        return len(self.imgs)
        
class DSB2018TestDataset(data.Dataset):

    def __init__(self, id_file, img_file, size_file, transform=None):
        id_f = open(id_file, 'r')
        self.ids = id_f.readlines()
        self.imgs = np.load(img_file)
        self.transform = transform
        
        self.sizes = []
        size_f = open(size_file, 'r')
        for line in size_f:
            self.sizes.append(line.split(' '))

    def __getitem__(self, idx):
        img = self.imgs[idx]
        id = self.ids[idx]
        size = self.sizes[idx]
            
        img = np.transpose(img, (2, 0 ,1)) # transpost to channel first
        
        img = torch.from_numpy(img).float()
        
        return id, img, size

    def __len__(self):
        return len(self.imgs)
