import torchvision.models as models
import torch
from torch import nn
from torch.nn import functional as F


class MeanPoolingMIL(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.embedding = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(.2)
        self.ln = nn.LayerNorm(1024)
        self.classifier = nn.Linear(1024, cfgs["num_classes"])
        
    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x = self.ln(x)
        y = self.classifier(x)
        # y = F.sigmoid(y)
        return y
    
    def inference(self, x):
        x = x.reshape([-1, 1024])
        y = self.forward(x)
        y = F.sigmoid(y)
        y = y.mean(dim=-2)
        pred = y.argmax(dim=-1)
        return pred
