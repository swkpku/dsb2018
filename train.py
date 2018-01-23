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

from dataset.dataset import DSB2018Dataset
from dataset.transforms import *
from models.unet import UNet
from models.losses import binary_cross_entropy2d
from models.metrics import runningScore
from trainer.trainer import get_trainer
from configs.lr_schedules import get_lr_schedule

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

TRAIN_DATA_ROOT = "/home/swk/dsb2018/stage1_train_data/"

def main(args):
    # preparing dataset and dataloader
    train_dataset = DSB2018Dataset(TRAIN_DATA_ROOT+'train_ids_train_256_'+str(args.dataset_index)+'.txt', 
                            TRAIN_DATA_ROOT+'X_train_256_'+str(args.dataset_index)+'.npy',
                            TRAIN_DATA_ROOT+'Y_train_256_'+str(args.dataset_index)+'.npy',
                            transform=Compose([
                                RandomRotate(10),                                        
                                RandomHorizontallyFlip()]))
        
    val_dataset = DSB2018Dataset(TRAIN_DATA_ROOT+'train_ids_val_256_'+str(args.dataset_index)+'.txt', 
                            TRAIN_DATA_ROOT+'X_val_256_'+str(args.dataset_index)+'.npy',
                            TRAIN_DATA_ROOT+'Y_val_256_'+str(args.dataset_index)+'.npy',
                            transform=Compose([
                                RandomRotate(10),                                        
                                RandomHorizontallyFlip()]))

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    # create log file
    num_train = train_dataloader.__len__()
    log_file = open(args.output_dir+str(args.model)+"_data"+str(args.dataset_index)+"_d6_u_c"+"_pre"+str(args.pretrained)+"_lr"+str(args.lr_schedule)+"_bs"+str(args.batch_size)+"_size"+str(args.img_size)+".log" ,"w")
    
    # training configuration
    config = {
        'train_batch_size': args.batch_size, 'val_batch_size': 10,
        'img_size': args.img_size,
        'arch': args.model, 'pretrained': args.pretrained, 'ckpt_title': "_data_"+str(args.dataset_index)+"_d6_u_c_lr"+str(args.lr_schedule)+"_bs"+str(args.batch_size)+"_size"+str(args.img_size),
        'lr_schedule_idx': args.lr_schedule, 'lr_schedule': get_lr_schedule(args.lr_schedule), 'weight_decay': 1e-5,
        'start_epoch': 0, 'epochs': args.num_epochs,
        'print_freq': args.print_freq,
        'log_file': log_file,
        'best_val_prec1': 0
    }

    # create model
    num_classes=1
    model = UNet(num_classes, in_channels=3, depth=6, start_filts=64, up_mode='upsample', merge_mode='concat')
    
    # create optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                          lr=config['lr_schedule'][0],
                                          weight_decay=config['weight_decay'])
    if (args.optimizer == 'SGD'):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                                          lr=config['lr_schedule'][0],
                                          momentum=config['momentum'],
                                          weight_decay=config['weight_decay'])

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
        print(config)
        
        config['log_file'] = open(args.output_dir+str(config['arch'])+"_lr"+str(config['lr_schedule_idx'])+"_bs"+str(config['train_batch_size'])+"_size"+str(config['img_size'])+".log" ,"a+")
        
        config['start_epoch'] = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    elif args.pretrained is True:
        print("using pretrained model")
        original_model = args.model.rsplit('_', 1)[0]
        pretrained_model = model_factory.create_model(original_model, num_classes=1000, pretrained=args.pretrained, test_time_pool=args.test_time_pool)
        
        pretrained_state = pretrained_model.state_dict()
        model_state = model.state_dict()

        fc_layer_name = 'fc'
        if args.model.startswith('dpn') or args.model.startswith('vgg'):
            fc_layer_name = 'classifier'
        
        for name, state in pretrained_state.items():
            if not name.startswith(fc_layer_name):
                model_state[name].copy_(state)
    else:
        print("no pretrained model or checkpoint loaded")

    if args.multi_gpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    # define loss function (criterion) & metrics
    # criterion =  torch.nn.BCELoss().cuda()
    metrics = runningScore(num_classes+1)

    # get trainer
    Trainer = get_trainer(train_dataloader, val_dataloader, model, optimizer, binary_cross_entropy2d, metrics, config)

    # run!
    Trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Inference')
    parser.add_argument('--output_dir', metavar='DIR', default='output/', help='path to output files')
    parser.add_argument('-d', '--dataset-index', default=0, type=int, metavar='N',help='index of dataset (default: 0)')
    parser.add_argument('--model', '-m', metavar='MODEL', default='unet', help='model architecture (default: unet)')
    parser.add_argument('--optimizer', '-opt', metavar='OPTIMIZER', default='Adam',help='optimizer (default: Adam)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',help='number of data loading workers (default: 0)')
    parser.add_argument('-lrs', '--lr-schedule', default=5, type=int, metavar='N', help='learning rate schedule (default: 5)')
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--img-size', default=256, type=int, metavar='N', help='Input image dimension')
    parser.add_argument('-e', '--num-epochs', default=5, type=int, metavar='N', help='Number of epochs')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--restore-checkpoint', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', default='False', action='store_true', help='use pre-trained model (default: True)')
    parser.add_argument('--multi-gpu', dest='multi_gpu', default='True', action='store_true', help='use multiple-gpus (default: True)')
    args = parser.parse_args()
    
    main(args)
