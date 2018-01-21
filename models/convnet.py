import torch
from torch import nn

dropout = torch.nn.Dropout(p=0.20)

def make_linear_bn_relu_dropout(in_channels, out_channels, p):
    return [
        nn.Linear(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(p)
    ]

class ConvRes(nn.Module):
    def __init__(self,insize, outsize):
        super(ConvRes, self).__init__()
        drate = .3
        self.math = nn.Sequential(
                 nn.BatchNorm2d(insize),
                 nn.Dropout(drate),
                 torch.nn.Conv2d(insize, outsize, kernel_size=2,padding=2),
                 nn.PReLU(),
                )
        
    def forward(self, x):
        return self.math(x) 

class ConvCNN(nn.Module):
    def __init__(self,insize, outsize, kernel_size=3, padding=1, pool=2, groups=1, stride=2, avg=True):
        super(ConvCNN, self).__init__()
        self.avg=avg
        self.math = torch.nn.Sequential(
            torch.nn.Conv2d(insize, outsize, kernel_size=kernel_size,padding=padding, groups=groups),
            torch.nn.BatchNorm2d(outsize),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(pool,stride),
        )
        self.avgpool=torch.nn.AvgPool2d(pool,pool)
        
    def forward(self, x):
        x=self.math(x)
        if self.avg is True:
            x=self.avgpool(x)
        return x   
        
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()        
        
        self.cnn1 = ConvCNN (3,64,  kernel_size=3, padding=0,  pool=3, avg=False)
        self.cnn2 = ConvCNN (64,128, kernel_size=3,padding=0,  pool=2,  avg=False)
        self.cnn3 = ConvCNN (128,128, kernel_size=3,padding=0,  pool=2, avg=False)
        self.cnn4 = ConvCNN (128,64, kernel_size=3,padding=0,  pool=2, avg=False)
        
        self.features = nn.Sequential( 
            self.cnn1,dropout,          
            self.cnn2,dropout,
            self.cnn3,dropout,
            self.cnn4,dropout,
        )
        
        self.fc1 = nn.Sequential(*make_linear_bn_relu_dropout(256, 512, 0.2))
        self.fc2 = nn.Sequential(*make_linear_bn_relu_dropout(512, 256, 0.2))
        
        self.classifier = torch.nn.Sequential(
            nn.Linear(256, 1),             
        )
        self.sig=nn.Sigmoid()        
            
    def forward(self, x):
        x = self.features(x) 
        x = x.view(x.size(0), -1)    
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifier(x)                
        x = self.sig(x)
        return x  
