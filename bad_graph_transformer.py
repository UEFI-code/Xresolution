import torch
import torch.nn as nn
import torch.nn.functional as F

class BadGraphTransformerDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=0, normalization = None, activation = nn.ReLU(), deepth = 2, debug=False):
        super(BadGraphTransformerDown, self).__init__()
        self.convEncodingGroupA = nn.Sequential()
        self.convEncodingGroupB = nn.Sequential()
        self.convEncodingGroupC = nn.Sequential()
        self.convDecodingGroup = nn.Sequential()
        for i in range(deepth):
            if i == 0:
                self.convEncodingGroupA.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
                self.convEncodingGroupA.append(activation)
                self.convEncodingGroupB.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
                self.convEncodingGroupB.append(activation)
                self.convEncodingGroupC.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
                self.convEncodingGroupC.append(activation)
            else:
                self.convEncodingGroupA.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
                self.convEncodingGroupA.append(activation)
                self.convEncodingGroupB.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
                self.convEncodingGroupB.append(activation)
                self.convEncodingGroupC.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
                self.convEncodingGroupC.append(activation)
            self.convDecodingGroup.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
            self.convDecodingGroup.append(activation)
        self.normalization = normalization
        self.debug = debug
    
    def forward(self, x):
        a, b, c = self.convEncodingGroupA(x), self.convEncodingGroupB(x), self.convEncodingGroupC(x)
        x = torch.matmul(a.transpose(2, 3), b) # Here is to semantic hybrid.
        if self.debug:
            print(f'Debug: xSqure shape {x.shape}')
        x = torch.matmul(x, c)
        x = self.convDecodingGroup(x)
        if self.normalization is not None:
            x = self.normalization(x)
        return x

class BadGraphTransformerUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=0, normalization = None, activation = nn.ReLU(), deepth = 2, debug=False):
        super(BadGraphTransformerUp, self).__init__()
        self.transConvEncodingGroupA = nn.Sequential()
        self.transConvEncodingGroupB = nn.Sequential()
        self.transConvEncodingGroupC = nn.Sequential()
        self.transConvDecodingGroup = nn.Sequential()
        for i in range(deepth):
            if i == 0:
                self.transConvEncodingGroupA.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
                self.transConvEncodingGroupA.append(activation)
                self.transConvEncodingGroupB.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
                self.transConvEncodingGroupB.append(activation)
                self.transConvEncodingGroupC.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
                self.transConvEncodingGroupC.append(activation)
            else:
                self.transConvEncodingGroupA.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding))
                self.transConvEncodingGroupA.append(activation)
                self.transConvEncodingGroupB.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding))
                self.transConvEncodingGroupB.append(activation)
                self.transConvEncodingGroupC.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding))
                self.transConvEncodingGroupC.append(activation)
            self.transConvDecodingGroup.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding))
            self.transConvDecodingGroup.append(activation)
        self.normalization = normalization
        self.debug = debug

    def forward(self, x):
        a, b, c = self.transConvEncodingGroupA(x), self.transConvEncodingGroupB(x), self.transConvEncodingGroupC(x)
        x = torch.matmul(a.transpose(2, 3), b) # Here is to semantic hybrid.
        if self.debug:
            print(f'Debug: xSqure shape {x.shape}')
        x = torch.matmul(x, c)
        x = self.transConvDecodingGroup(x)
        if self.normalization is not None:
            x = self.normalization(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64)
    badGraphTransDown = BadGraphTransformerDown(3, 16, debug=True)
    x = badGraphTransDown(x)
    print(f'After badGraphTransDown: {x.shape}')
    badGraphTransUp = BadGraphTransformerUp(16, 16, debug=True)
    x = badGraphTransUp(x)
    print(f'After badGraphTransUp: {x.shape}')