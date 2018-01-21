from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import numpy as np
import pandas as pd

class DSB2018Dataset(data.Dataset):

    def __init__(self, id_file, img_file, mask_file=None, transform=None):
        id_f = open(id_file, 'r')
        self.ids = id_f.readlines()
        self.imgs = np.load(img_file)
        self.masks = None
        
        if (mask_file):
            self.masks = np.load(mask_file)
            
        self.transform = transform

    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = np.transpose(img, (2, 0 ,1)) # transpost to channel first
        id = self.ids[idx]
        print(img.shape)
        print(id)
        
        if self.transform:
            img = self.transform(img)
        
        if (self.masks is not None):
            mask = self.masks[idx]
            return id, img, mask

        return id, img

    def __len__(self):
        return len(self.imgs)
