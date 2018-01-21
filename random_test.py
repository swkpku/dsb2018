from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import torch.utils.data as data

class Dataset(data.Dataset):

    def __init__(self, data_json, with_label, transform=None):
        self.data_df = pd.read_json(data_json)
        self.with_label = with_label
        self.transform = transform
        
        # concat bands into (N,2,75,75) images
        band_1 = np.concatenate([img for img in self.data_df['band_1']]).reshape(-1, 75, 75)
        band_2 = np.concatenate([img for img in self.data_df['band_2']]).reshape(-1, 75, 75)
        self.full_img = np.stack([band_1, band_2], axis=1)
        
        tmp = [label for label in self.data_df['is_iceberg']]
        print(band_1[5].astype(np.float32).dtype)
        labels = [label for label in self.data_df['is_iceberg']]
        
train_set = Dataset('/media/swk/data/iceberg/data/train_1.json', with_label=True)
