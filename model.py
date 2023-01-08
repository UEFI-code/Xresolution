import torch
import torch.nn as nn
import torch.nn.functional as F
import bad_graph_transformer

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.relu = nn.ReLU()
        self.up1 = bad_graph_transformer.BadGraphTransformerUp(3, 8, 3, stride=2, padding=0, normalization=nn.BatchNorm2d(8), activation=self.relu, dropout=None)
        self.up2 = bad_graph_transformer.BadGraphTransformerUp(8, 8, 3, stride=2, padding=0, normalization=nn.BatchNorm2d(8), activation=self.relu, dropout=None)
        #self.up3 = bad_graph_transformer.BadGraphTransformerUp(8, 8, 3, stride=2, padding=0, normalization=nn.BatchNorm2d(8), activation=self.relu, dropout=None)
        self.down = bad_graph_transformer.BadGraphTransformerDown(8, 3, 3, stride=2, padding=0, normalization=nn.BatchNorm2d(3), activation=self.relu, dropout=None)
    
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        #x = self.up3(x)
        x = self.down(x)
        return x
    