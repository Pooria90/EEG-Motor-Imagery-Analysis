'''
This modules contains the models that I implemented,
and the block the make up this models
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
    
# EEGNet
class EEGNet(nn.Module):
    def __init__(self, F1 = 8, C = 22, D = 2, F2 = 16, kernel_length = 125):
        super().__init__()
        self.name = 'EEGNet'
        self.F1 = F1
        self.C = C
        self.D = D
        self.F2 = F2
        self.L = kernel_length
        
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
            LinearConstrained(240,4,max_norm = 0.25,bias = False)
        )
        
    def forward(self, xb):
        xb = self.layer1(xb)
        xb = self.layer2(xb)
        xb = self.layer3(xb)
        xb = F.log_softmax(self.layer4(xb),dim=1)
        return xb

# Shallow ConvNet
class ShallowNet(nn.Module):
    def __init__(self, C=22, F1=40):
        super().__init__()
        self.C  = C
        self.F1 = F1
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
            LinearConstrained(2600, 4, max_norm=0.5, bias=False)
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
        self.fc_units = ((((self.T - self.ks+1)//2 - self.ks+1)//2 - self.ks+1)//2 - self.ks+1)//2 * self.F4
        self.fc = nn.Sequential(
            LinearConstrained(self.fc_units, self.N, max_norm=0.5, bias=False)
        )

    def forward(self, xb):
        xb = self.layer1(xb)
        xb = self.layer2(xb)
        xb = self.layer3(xb)
        xb = self.layer4(xb)
        xb = F.log_softmax(self.fc(xb), dim=1)
        return xb
        
