import time
import torch
import csv
import numpy as np
#import visdom
        
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
        
        outfile = open("stage2/" + self.config['pred_filename'], "w")
        
        # prediction
        print("start prediction")
        end = time.time()
        for i, (imgs, targets, _) in enumerate(self.test_dataloader):
            # measure data loading time
            data_time = time.time() - end
			
            input_var = torch.autograd.Variable(imgs, volatile=True)

            # compute output
            output = self.model(input_var)
            probs, predicts = torch.topk(output.data, 10)

            for idx, target in enumerate(targets):
                outfile.write("%d " % target)
                for prob, predict in zip(probs[idx], predicts[idx]):
                    outfile.write("%d:%f " % (predict, prob))
                outfile.write("\n")
            
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