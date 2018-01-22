import time
import torch
import csv
import numpy as np
from PIL import Image

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
        
class Predictor():
    def __init__(self, test_dataloader, model, config):
        self.test_dataloader = test_dataloader
        self.model = model
        self.config = config
        
        # visualizer
        #self.viz = visdom.Visdom()
        #self.iter_viz = self.viz.line(Y=np.array([0]), X=np.array([0]))
        
    def run(self):
        torch.backends.cudnn.benchmark = True # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
                                              # If this is set to false, uses some in-built heuristics that might not always be fastest.
        
        # switch to evaluate mode
        self.model.eval()
        
        outfile = open(self.config['pred_filename'], "w")
        
        # prediction
        print("start prediction")
        end = time.time()
        for i, (id, imgs) in enumerate(self.test_dataloader):
            # measure data loading time
            data_time = time.time() - end
            
            input_var = torch.autograd.Variable(imgs, volatile=True)

            # compute output
            output = self.model(input_var)
            
            predicts = output.data.cpu().numpy()
            predicts_t = (predicts > 0.5).astype(np.uint8)
            
            # Create list of upsampled test masks
            preds_test_upsampled = []
            for i in range(len(predicts)):
                preds_test_upsampled.append(resize(np.squeeze(predicts[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))
            
            #img = np.transpose(imgs.numpy()[0], (1,2,0)).astype(np.uint8)
            #print(img.shape)
            #print(output.data.cpu().numpy().astype(np.uint8)[0][0].shape)
            
            #img = Image.fromarray(img, mode='RGB')
            #img.show()
            
            #mask = Image.fromarray(output.data.cpu().numpy().astype(np.uint8)[0][0], mode='L')
            #mask.show()
            
            #exit()
            
            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()
            
            # visulization
            #self.viz.line(
            #    Y=np.array([(i+1)*self.config['test_batch_size']]),
            #    X=np.array([i+1]),
            #    win=self.iter_viz,
            #    update="append"
            #)

            if i % self.config['print_freq'] == 0:
                print('Iter: [{0}/{1}]\t'
                      'Time {batch_time:.3f}\t'
                      'Data {data_time:.3f}\t'.format(
                       i, len(self.test_dataloader), batch_time=batch_time,
                       data_time=data_time))
                
        outfile.close()
        
    
def get_predictor(test_dataloader, model, config):
    return Predictor(test_dataloader, model, config)
