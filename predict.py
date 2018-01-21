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

import model_factory
from dataset import Dataset

from predictor.predictor_singlecrop import get_predictor

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch Inference')
parser.add_argument('--output_dir', metavar='DIR', default='./',
					help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
					help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
					help='number of data loading workers (default: 1)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
					metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=75, type=int,
					metavar='N', help='Input image dimension')
parser.add_argument('--print-freq', '-p', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--restore-checkpoint', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', default='True', action='store_true',
					help='use pre-trained model (default: True)')
parser.add_argument('--multi-gpu', dest='multi_gpu', default='True', action='store_true',
					help='use multiple-gpus (default: True)')
parser.add_argument('--no-test-pool', dest='test_time_pool', action='store_false',
					help='use pre-trained model')
                    
args = parser.parse_args()

transforms = model_factory.get_transforms_iceberg(
    args.model,
    args.img_size)

config = {
    'test_batch_size': 100,
    'checkpoint': 'checkpoints/ConvNet_sigmoid__lr3_bs32_size75_epoch_7_iter_40.pth.tar',
    'print_freq': 10,
    'pred_filename': "predicts/ConvNet_sigmoid__lr3_bs32_size75_epoch_7_iter_40_singlecrop.csv",
	'arch': args.model
}

data_path = "/home/iceberg/"

num_classes = 2

# get dataset
print('getting dataset...')

test_dataset = Dataset(
		data_json=data_path+"test_1.json",
		with_label=False,
		transform=None)
        
# get data loader
print('getting data loader...')

test_dataloader = data.DataLoader(
		test_dataset,
		batch_size=config["test_batch_size"], shuffle=False,
		num_workers=args.workers, pin_memory=True)

# define model
model = model_factory.create_model(args.model, num_classes=num_classes, pretrained=False)
model = torch.nn.DataParallel(model).cuda()

# load checkpoint
if not os.path.isfile(config['checkpoint']):
    print("=> no checkpoint found at '{}'".format(config['checkpoint']))
    
print("=> loading checkpoint '{}'".format(config['checkpoint']))
checkpoint = torch.load(config['checkpoint'])
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint:")

print('Epoch: [{0}][{1}]\t'
      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
       checkpoint['epoch'], checkpoint['iter'], loss=checkpoint['loss'], top1=checkpoint['top1'], top5=checkpoint['top5']))

#del checkpoint # save some GPU memory

# get trainer
Predictor = get_predictor(test_dataloader, model, config)

# Run!
Predictor.run()
