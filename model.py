import torch
import torch.nn as nn
import torch.nn.functional as F
import bad_graph_transformer

class myModel(nn.Module):
    def __init__(self, embeddingDim = 8, embeddingDeepth = 2, deepth = 3, debug=False):
        super(myModel, self).__init__()
        self.embedingGroup = nn.Sequential()
        for i in range(deepth):
            if i == 0:
                self.embedingGroup.append(bad_graph_transformer.BadGraphTransformerUp(3, embeddingDim, 3, stride=2, padding=0, deepth=embeddingDeepth, debug=debug))
            else:
                self.embedingGroup.append(bad_graph_transformer.BadGraphTransformerUp(embeddingDim, embeddingDim, 3, stride=2, padding=0, deepth=embeddingDeepth, debug=debug))
            self.embedingGroup.append(bad_graph_transformer.BadGraphTransformerDown(embeddingDim, embeddingDim, 3, stride=2, padding=0, deepth=embeddingDeepth, debug=debug))
        
        self.winduper = nn.Sequential(
            bad_graph_transformer.BadGraphTransformerUp(embeddingDim, embeddingDim, 3, stride=2, padding=0, deepth=embeddingDeepth, debug=debug),
            nn.Conv2d(embeddingDim, 3, 5, stride=3, padding=0) # A normal convolutional decoder is enough.
        )

        self.debug = debug
    
    def forward(self, x):
        if self.debug:
            i = 0
            for layer in self.embedingGroup:
                x = layer(x)
                print(f'Embedding layer {i} output shape: {x.shape}')
                i += 1
            i = 0
            for layer in self.winduper:
                x = layer(x)
                print(f'Winduper layer {i} output shape: {x.shape}')
        else:
            x = self.embedingGroup(x)
            x = self.winduper(x)
        return x

if __name__ == '__main__':
    model = myModel(embeddingDim=1, debug=True)
    x = torch.rand(1, 3, 256, 256)
    y = model(x)
    print(y.shape)