import time
import torch
import csv
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize
from skimage.morphology import label
import torch.nn.functional as F

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
        
        new_test_ids = []
        rles = []
        
        # prediction
        print("start prediction")
        end = time.time()
        for idx, (id, imgs, size) in enumerate(self.test_dataloader):
            # measure data loading time
            data_time = time.time() - end
            
            input_var = torch.autograd.Variable(imgs, volatile=True)

            # compute output
            output = self.model(input_var)
            output = F.sigmoid(output)
            
            predicts = output.data.cpu().numpy()
            predicts_t = (predicts > 0.5).astype(np.uint8)
            
            # Create list of upsampled test masks
            preds_test_upsampled = []
            for i in range(len(predicts)):
                preds_test_upsampled.append(resize(np.squeeze(predicts[i]), 
                                       (int(size[0][i]), int(size[1][i])), 
                                       mode='constant', preserve_range=True))
            
            for n, id_ in enumerate(id):
                rle = list(prob_to_rles(preds_test_upsampled[n]))
                rles.extend(rle)
                new_test_ids.extend([id_] * len(rle))
            
            #img = np.transpose(imgs.numpy()[0], (1,2,0)).astype(np.uint8)
            #print(img.shape)
            #print(output.data.cpu().numpy().astype(np.uint8)[0][0].shape)
            
            #img = Image.fromarray(img, mode='RGB')
            #img.show()
            
            #for i in range(10):
            #    mask = Image.fromarray(output.data.cpu().numpy().astype(np.uint8)[i][0], mode='L')
            #    mask.show()
            
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

            if idx % self.config['print_freq'] == 0:
                print('Iter: [{0}/{1}]\t'
                      'Time {batch_time:.3f}\t'
                      'Data {data_time:.3f}\t'.format(
                       idx, len(self.test_dataloader), batch_time=batch_time,
                       data_time=data_time))
                
        sub = pd.DataFrame()
        sub['ImageId'] = new_test_ids
        sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
        sub.to_csv(self.config['pred_filename'], index=False)
    
def get_predictor(test_dataloader, model, config):
    return Predictor(test_dataloader, model, config)
