import torch
import torch.nn as nn
import torch.nn.functional as F

class BadGraphTransformerDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=0, normalization = None, activation = nn.ReLU(), debug=False):
        super(BadGraphTransformerDown, self).__init__()
        self.convA = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.convB = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.convC = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation
        self.normalization = normalization
        self.debug = debug
    
    def forward(self, x):
        a, b, c = self.convA(x), self.convB(x), self.convC(x)
        a, b, c = self.activation(a), self.activation(b), self.activation(c)
        x = torch.matmul(a.transpose(2, 3), b) # Here is to semantic hybrid.
        if self.debug:
            print(f'Debug: {x.shape}')
        x = torch.matmul(x, c)
        if self.normalization is not None:
            x = self.normalization(x)
        return x

class BadGraphTransformerUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=0, normalization = None, activation = nn.ReLU(), debug=False):
        super(BadGraphTransformerUp, self).__init__()
        self.transConvA = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.transConvB = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.transConvC = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation
        self.normalization = normalization
        self.debug = debug

    def forward(self, x):
        a, b, c = self.transConvA(x), self.transConvB(x), self.transConvC(x)
        a, b, c = self.activation(a), self.activation(b), self.activation(c)
        x = torch.matmul(a.transpose(2, 3), b) # Here is to semantic hybrid.
        if self.debug:
            print(f'Debug: {x.shape}')
        x = torch.matmul(x, c)
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