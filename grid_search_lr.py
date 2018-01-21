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

from trainer.trainer import get_trainer
from configs.lr_schedules import get_lr_schedule


parser = argparse.ArgumentParser(description='PyTorch Inference')
parser.add_argument('--output_dir', metavar='DIR', default='search_lr_output/',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 6)')
parser.add_argument('-lrs', '--lr-schedule', default=1, type=int,
                    metavar='N', help='learning rate schedule (default: 1)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--img-size', default=75, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('-e', '--num-epochs', default=5, type=int,
                    metavar='N', help='Number of epochs')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--restore-checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', default='False', action='store_true',
                    help='use pre-trained model (default: True)')
parser.add_argument('--multi-gpu', dest='multi_gpu', default='True', action='store_true',
                    help='use multiple-gpus (default: True)')
parser.add_argument('--no-test-pool', dest='test_time_pool', action='store_false',
                    help='use pre-trained model')
                    

data_path = '/media/swk/data/iceberg/data/'
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def main():
    args = parser.parse_args()
    
    transforms = model_factory.get_transforms_eval(
        args.model,
        args.img_size)
        
    train_dataset = Dataset(
        data_json=data_path+"train_1.json",
        with_label=True,
        transform=None)
        
    val_dataset = Dataset(
        data_json=data_path+"val_1.json",
        with_label=True,
        transform=None)

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
        
    num_train = train_dataloader.__len__()
    
    log_file = open(args.output_dir+str(args.model)+"_grid_search_lr"+"_bs"+str(args.batch_size)+"_size"+str(args.img_size)+".log" ,"a+")
        
    # configuration
    config = {
        'train_batch_size': args.batch_size, 'val_batch_size': 100,
        'arch': args.model, 'pretrained': args.pretrained,
        'optimizer': 'Adam', 'lr_schedule_idx': args.lr_schedule, 'lr_schedule': get_lr_schedule(args.lr_schedule), 'weight_decay': 0,
        'resume': None,
        'start_epoch': 0, 'epochs': 150,
        'print_freq': args.print_freq, 'validate_freq': num_train-1, 'save_freq': num_train+1,
        'log_file': log_file
    }

    lr_list = [[1e-2], [9e-3], [8e-3], [7e-3], [6e-3], [5e-3], [4e-3], [3e-3], [2e-3], [1e-3]]

    for lr in lr_list:

        print("\ntest lr" + str(lr), file=config['log_file'], flush=True)
        config["lr_schedule"] = lr
        
        # create model
        num_classes = 1
        model = model_factory.create_model(args.model, num_classes=num_classes, pretrained=False, test_time_pool=args.test_time_pool)

        # resume from a checkpoint
        if args.restore_checkpoint and os.path.isfile(args.restore_checkpoint):
            print("=> loading checkpoint '{}'".format(args.restore_checkpoint))
            checkpoint = torch.load(args.restore_checkpoint)
        
            print('Epoch: [{0}] iter: [{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                     checkpoint['epoch'], checkpoint['iter'],
                     loss=checkpoint['loss'],
                      top1=checkpoint['top1'],
                   top5=checkpoint['top5']))

            config = checkpoint['config']
            # set to resume mode
            config['resume'] = args.restore_checkpoint
            print(config)
        
            config['log_file'] = open(args.output_dir+str(config['arch'])+"_lr"+str(config['lr_schedule_idx'])+"_bs"+str(config['train_batch_size'])+"_size"+str(config['img_size'])+".log" ,"a+")
        elif args.pretrained == True:
            print("using pretrained model")
            original_model = args.model.split('_')[0]
            pretrained_model = model_factory.create_model(original_model, num_classes=1000, pretrained=args.pretrained, test_time_pool=args.test_time_pool)
        
            pretrained_state = pretrained_model.state_dict()
            model_state = model.state_dict()

            fc_layer_name = 'fc'
            if args.model.startswith('dpn'):
                fc_layer_name = 'classifier'
        
            for name, state in pretrained_state.items():
                if not name.startswith(fc_layer_name):
                    model_state[name].copy_(state)
        else:
            print("please use pretrained model")

            if args.multi_gpu:
                model = torch.nn.DataParallel(model).cuda()
            else:
                model = model.cuda()

            # define loss function (criterion) and optimizer
            criterion = torch.nn.BCELoss().cuda()

            # get trainer
            Trainer = get_trainer(train_dataloader, val_dataloader, model, criterion, config)

            # Run!
            Trainer.run()


if __name__ == '__main__':
    main()
