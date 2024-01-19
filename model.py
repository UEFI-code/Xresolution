import torch
import torch.nn as nn
import torch.nn.functional as F
import bad_graph_transformer

class myModel(nn.Module):
    def __init__(self, embeddingDim = 8, embeddingDeepth = 2, debug=False):
        super(myModel, self).__init__()
        self.relu = nn.ReLU()
        self.up1 = bad_graph_transformer.BadGraphTransformerUp(3, embeddingDim, 3, stride=2, padding=0, deepth=embeddingDeepth, debug=debug)
        self.down1 = bad_graph_transformer.BadGraphTransformerDown(embeddingDim, embeddingDim, 3, stride=2, padding=0, deepth=embeddingDeepth, debug=debug)
        self.up2 = bad_graph_transformer.BadGraphTransformerUp(embeddingDim, embeddingDim, 3, stride=2, padding=0, deepth=embeddingDeepth, debug=debug)
        self.down2 = bad_graph_transformer.BadGraphTransformerDown(embeddingDim, 3, 3, stride=2, padding=0, deepth=1, debug=debug)
        self.debug = debug
    
    def forward(self, x):
        x = self.up1(x)
        if self.debug:
            print(f'x.shape after up1: {x.shape}')
        x = self.down1(x)
        if self.debug:
            print(f'x.shape after down1: {x.shape}')
        x = self.up2(x)
        if self.debug:
            print(f'x.shape after up2: {x.shape}')
        x = self.down2(x)
        return x

if __name__ == '__main__':
    model = myModel(embeddingDim=1, debug=True)
    x = torch.rand(1, 3, 256, 256)
    y = model(x)
    print(y.shape)