'''
This modules contains the models that I implemented,
and the blocks the make up these models
'''

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# a block that enables us to enter customized functions in the structure of an nn.Module
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)

# a function for adding Flatten layers to Conv2d architectures
def myreshape(xb):
    return xb.view(-1,xb.shape[1]*xb.shape[3])


# constrained blocks are required for implementing of EEGNet and ShallowConvNet
class Conv2dConstrained(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dConstrained, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p = 2, dim = 0, maxnorm = self.max_norm
        )
        return super(Conv2dConstrained, self).forward(x)

class LinearConstrained(nn.Linear):
    def __init__(self, *args, max_norm = 0.25, **kwargs):
        self.max_norm = max_norm
        super(LinearConstrained, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p = 2, dim = 0, maxnorm = self.max_norm
        )
        return super(LinearConstrained, self).forward(x)
    
# a class that allows us to define linear layers without specifying in_features
class LinearModified(nn.Module):
    def __init__(self, out_features, bias=False, max_norm=None):
        super().__init__()
        self.in_features = None
        self.out_features = out_features
        self.bias = bias
        self.max_norm = max_norm
        self.__built = False
        self.lin = 0
        
    def forward(self, xb):
        assert xb.ndim == 2, 'xb should have 2 dimensions'
        if self.__built == False:
            self.__built = True
            self.in_features = xb.shape[1]
            dev = 'cpu' if xb.get_device == -1 else 'cuda'
            if self.max_norm == None:
                self.lin = nn.Linear(self.in_features, self.out_features, bias=self.bias).to(dev)
            else:
                self.lin = LinearConstrained(self.in_features, self.out_features, max_norm=self.max_norm, bias=self.bias).to(dev)
        xb = self.lin(xb)
        return xb
    
# EEGNet
class EEGNet(nn.Module):
    def __init__(self, F1 = 8, C = 22, D = 2, F2 = 16, ks = 125, N = 4):
        super().__init__()
        self.name = 'EEGNet'
        self.F1 = F1
        self.C  = C
        self.D  = D
        self.F2 = F2
        self.L  = ks
        self.N  = N
        
        # Conv2D layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, self.F1, kernel_size = (1,self.L),bias = False, padding = (0, self.L//2)),
            nn.BatchNorm2d(self.F1)
        )
        
        # Depthwise Conv2D layer
        self.layer2 = nn.Sequential(
            Conv2dConstrained(self.F1, self.F1*self.D, kernel_size = (self.C,1), max_norm = 1, bias = False, groups = self.F1, padding = (0,0)),
            nn.BatchNorm2d(self.F1*self.D),
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(p = 0.25)
        )
        
        # Seperable Conv2D layer
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.F1*self.D, self.F1*self.D, kernel_size = (1,self.L//4), bias = False, groups = self.F1*self.D, padding = (0,self.L//8)),
            nn.Conv2d(self.F1*self.D, self.F2, kernel_size = (1,1), stride = 1, bias = False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1,8)),
            nn.Dropout(p = 0.25),
            Lambda(myreshape),
        )
        
        # Classification layer
        self.layer4 = nn.Sequential(
            #LinearConstrained(240,4,max_norm = 0.25,bias = False)
            LinearModified(self.N, max_norm=0.25)
        )
        
    def forward(self, xb):
        xb = self.layer1(xb)
        xb = self.layer2(xb)
        xb = self.layer3(xb)
        xb = F.log_softmax(self.layer4(xb),dim=1)
        return xb

# Shallow ConvNet
class ShallowNet(nn.Module):
    def __init__(self, C=22, F1=40, N=4):
        super().__init__()
        self.C  = C
        self.F1 = F1
        self.N  = N
        self.conv = nn.Sequential(
            Conv2dConstrained(1, self.F1, kernel_size=(1,14), max_norm=2, bias=False, padding=(0,0)),
            Conv2dConstrained(self.F1, self.F1, kernel_size=(self.C,1), max_norm=2, bias=False, groups=self.F1, padding=(0,0)),
            nn.BatchNorm2d(self.F1),
            Lambda(torch.square),
            nn.AvgPool2d((1,35),stride=(1,7)),
            Lambda(torch.log),
            Lambda(myreshape),
            nn.Dropout(p=0.5)
        )
        self.fc = nn.Sequential(
            #LinearConstrained(2600, 4, max_norm=0.5, bias=False)
            LinearModified(self.N, max_norm=0.5)
        )
        
    def forward(self, xb):
        xb = self.conv(xb)
        xb = F.log_softmax(self.fc(xb),dim=1)
        return xb

# Deep ConvNet
class DeepNet(nn.Module):
    def __init__(self, in_shape=(22,500), F1=25, F2=50, F3=100, F4=200, ks=5, N=4):
        super().__init__()
        self.C  = in_shape[0]
        self.T  = in_shape[1]
        self.F1 = F1
        self.F2 = F2
        self.F3 = F3
        self.F4 = F4
        self.ks = ks
        self.N  = N
        self.layer1 = nn.Sequential(
            Conv2dConstrained(1, self.F1, kernel_size=(1,self.ks), max_norm=2),
            Conv2dConstrained(self.F1, self.F1, kernel_size=(self.C,1), max_norm=2, groups=self.F1),
            nn.BatchNorm2d(self.F1),
            nn.ELU(),
            nn.MaxPool2d((1,2)),
            nn.Dropout(p=0.5)
        )
        self.layer2 = nn.Sequential(
            Conv2dConstrained(self.F1, self.F2, kernel_size=(1,self.ks), max_norm=2),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.MaxPool2d((1,2)),
            nn.Dropout(p=0.5)
        )
        self.layer3 = nn.Sequential(
            Conv2dConstrained(self.F2, self.F3, kernel_size=(1,self.ks), max_norm=2),
            nn.BatchNorm2d(self.F3),
            nn.ELU(),
            nn.MaxPool2d((1,2)),
            nn.Dropout(p=0.5)
        )
        self.layer4 = nn.Sequential(
            Conv2dConstrained(self.F3, self.F4, kernel_size=(1,self.ks), max_norm=2),
            nn.BatchNorm2d(self.F4),
            nn.ELU(),
            nn.MaxPool2d((1,2)),
            nn.Dropout(p=0.5),
            Lambda(myreshape)
        )
        #self.fc_units = ((((self.T - self.ks+1)//2 - self.ks+1)//2 - self.ks+1)//2 - self.ks+1)//2 * self.F4
        self.fc = nn.Sequential(
            #LinearConstrained(self.fc_units, self.N, max_norm=0.5, bias=False)
            LinearModified(self.N, max_norm=0.5)
        )

    def forward(self, xb):
        xb = self.layer1(xb)
        xb = self.layer2(xb)
        xb = self.layer3(xb)
        xb = self.layer4(xb)
        xb = F.log_softmax(self.fc(xb), dim=1)
        return xb

# CNN-i structures
# Based on Deep Learning for EEG motor imagery classification based on multi-layer CNNs feature fusion
class CNN_1(nn.Module):
    def __init__(self,ks1=30,f1=50,f2=50,dense=512,N=4,p=0.5):
        super().__init__()
        self.D = dense
        self.ks1 = ks1
        self.f1 = f1
        self.f2 = f2
        self.N = N
        self.layer1 = nn.Sequential(
            Conv2dConstrained(1,self.f1,kernel_size=(1,self.ks1),max_norm=2,bias=False),
            nn.BatchNorm2d(self.f1),
            Conv2dConstrained(self.f1,self.f2,kernel_size=(22,1),max_norm=2,groups=50,bias=False),
            nn.BatchNorm2d(self.f2),
            nn.ELU(),
            nn.MaxPool2d((1,3),stride=3),
            nn.Dropout(p),
            Lambda(myreshape)
        )
        self.layer2 = nn.Sequential(
            nn.LinearModified(self.D,bias=False),
            nn.BatchNorm1d(self.D),
            nn.ELU(),
            nn.Dropout(p)
        )
        self.layer3 = nn.Sequential(
            LinearConstrained(self.D,self.N,max_norm=0.5,bias=False)
        )

    def forward(self, xb):
        xb = self.layer1(xb)
        xb = self.layer2(xb)
        xb = F.log_softmax(self.layer3(xb),dim=1)
        return xb