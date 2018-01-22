"""Sample PyTorch Inference script
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import torch
import torch.autograd as autograd
import torch.utils.data as data

from dataset.dataset import DSB2018TestDataset
from models.unet import UNet
from predictor.predictor_singlecrop import get_predictor

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch Inference')
parser.add_argument('--output_dir', metavar='DIR', default='./',
					help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='unet',
					help='model architecture (default: unet)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
					help='number of data loading workers (default: 0)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
					metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=256, type=int,
					metavar='N', help='Input image dimension')
parser.add_argument('--print-freq', '-p', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
                    
args = parser.parse_args()

config = {
    'test_batch_size': 100,
    'checkpoint': 'checkpoints/unet_lr5_bs16_size256_epoch_4.pth.tar',
    'print_freq': 10,
    'pred_filename': "predicts/unet_lr5_bs16_size256_epoch_4.csv",
	'arch': args.model
}

TEST_DATA_ROOT = "/home/swk/dsb2018/stage1_test_data/"

num_classes = 2

# get dataset
print('getting dataset...')

test_dataset = DSB2018TestDataset(TEST_DATA_ROOT+'test_ids_256.txt', 
                            TEST_DATA_ROOT+'X_test_256.npy')
        
# get data loader
print('getting data loader...')

test_dataloader = data.DataLoader(
		test_dataset,
		batch_size=config["test_batch_size"], shuffle=False,
		num_workers=args.workers, pin_memory=True)

# define model
num_classes=2
model = UNet(num_classes, in_channels=3, depth=5, merge_mode='concat')
model = torch.nn.DataParallel(model).cuda()

# load checkpoint
if not os.path.isfile(config['checkpoint']):
    print("=> no checkpoint found at '{}'".format(config['checkpoint']))
    
print("=> loading checkpoint '{}'".format(config['checkpoint']))
checkpoint = torch.load(config['checkpoint'])
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint:")

print('Epoch: [{0}]\t'
      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
       checkpoint['epoch'], loss=checkpoint['loss']))

#del checkpoint # save some GPU memory

# get trainer
Predictor = get_predictor(test_dataloader, model, config)

# Run!
Predictor.run()
