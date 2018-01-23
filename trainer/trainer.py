import time, os
import torch
import shutil
import numpy as np

import matplotlib.pyplot as plt

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class Trainer():
    def __init__(self, train_dataloader, val_dataloader, model, optimizer, criterion, metrics, config):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.config = config

        self.train_batch_time = AverageMeter()
        self.train_data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.train_top1 = AverageMeter()
        self.train_top5 = AverageMeter()
            
        self.decay_rate = 0

    def run(self):
        torch.backends.cudnn.benchmark = True # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
                                              # If this is set to false, uses some in-built heuristics that might not always be fastest.

        # train
        print("start training", file=self.config['log_file'], flush=True)
        for epoch in range(self.config['start_epoch'], self.config['epochs']):
            # learning rate schedule
            self._adjust_learning_rate(epoch, self.config['lr_schedule'])
            
            # train for one epoch
            self._train(epoch)
            
            # save checkpoint
            tmp = self.config['log_file']
            self.config['log_file'] = None
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_val_prec1': self.config['best_val_prec1'],
                'optimizer' : self.optimizer.state_dict(),
                'loss' : self.train_losses,
                'top1' : self.train_top1,
                'top5' : self.train_top5,
                'config': self.config
            }, is_best=False)
            self.config['log_file'] = tmp
            
            # validate
            self._validate(self.val_dataloader)

        
    def _train(self, epoch):
        # switch to train mode
        self.model.train()
        
        end = time.time()
        for i, (_, img, target) in enumerate(self.train_dataloader):
            # measure data loading time
            self.train_data_time.update(time.time() - end)
            
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(img)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(input=output, target=target_var)
        
            # measure accuracy and record loss
            self.train_losses.update(loss.data[0], img.size(0))

            # compute gradient and do one optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            self.train_batch_time.update(time.time() - end)
            end = time.time()

            if i % self.config['print_freq'] == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       epoch, i, len(self.train_dataloader), batch_time=self.train_batch_time,
                       data_time=self.train_data_time, loss=self.train_losses), file=self.config['log_file'], flush=True)

    def _validate(self, val_loader):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, (_, img, target) in enumerate(self.val_dataloader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(img, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = self.model(input_var)                
            loss = self.criterion(input=output, target=target_var)

            losses.update(loss.data[0], img.size(0))
            
            pred = output.data.max(1)[1].cpu().numpy()
            gt = target_var.data.cpu().numpy()
            self.metrics.update(gt, pred)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.config['print_freq'] == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5), file=self.config['log_file'], flush=True)
                       
        score, class_iou = self.metrics.get_scores()
        for k, v in score.items():
            print(k, v, file=self.config['log_file'], flush=True)
        self.metrics.reset()
        print('* Test:\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
               batch_time=batch_time, loss=losses), file=self.config['log_file'], flush=True)



    def _save_checkpoint(self, state, is_best):
        epoch = state['epoch']
        filename = 'checkpoints/' + self.config['arch'] + '_' + self.config['ckpt_title'] + '_epoch_' + str(epoch) + '.pth.tar'
        #torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, self.config['arch'] + '_epoch_' + str(epoch) + '_iter_' + str(iteration) + '_model_best.pth.tar')

        
    def _adjust_learning_rate(self, epoch, lr_schedule):
        """Sets the learning rate according to the learning rate schedule"""
        idx = epoch
        if idx >= len(lr_schedule):
           idx = len(lr_schedule) - 1
        lr = lr_schedule[idx]
        print('learning rate = %f' % lr, file=self.config['log_file'], flush=True)
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
          
    def _accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def get_trainer(train_dataloader, val_dataloader, model, optimizer, criterion, metrics, config):
    return Trainer(train_dataloader, val_dataloader, model, optimizer, criterion, metrics, config)
