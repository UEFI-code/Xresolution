import torch
import torch.nn as nn
import torch.nn.functional as F

class BadGraphTransformerDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=0, normalization = None, activation = None, dropout = None):
        super(BadGraphTransformerDown, self).__init__()
        self.convA = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.convB = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.convC = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation
        self.normalization = normalization
        self.dropout = dropout
    
    def forward(self, x):
        a = self.convA(x)
        b = self.convB(x)
        c = self.convC(x)
        if self.activation is not None:
            a = self.activation(a)
            b = self.activation(b)
            c = self.activation(c)
        x = a * b + c
        if dropout is not None:
            x = self.dropout(x)
        if self.normalization is not None:
            x = self.normalization(x)
        return x

class BadGraphTransformerUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=0, normalization = None, activation = None, dropout = None):
        super(BadGraphTransformerUp, self).__init__()
        self.transConvA = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.transConvB = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.transConvC = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation
        self.normalization = normalization
        self.dropout = dropout

    def forward(self, x):
        a = self.transConvA(x)
        b = self.transConvB(x)
        c = self.transConvC(x)
        if self.activation is not None:
            a = self.activation(a)
            b = self.activation(b)
            c = self.activation(c)
        x = a * b + c
        if self.dropout is not None:
            x = self.dropout(x)
        if self.normalization is not None:
            x = self.normalization(x)
        return x
    
    

        
        