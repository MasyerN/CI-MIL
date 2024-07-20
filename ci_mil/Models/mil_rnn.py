from torch.autograd import forward_ad
import torchvision.models as models
import torch
from torch import nn
from torch.nn import functional as F
import time


class MILRNN(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.encoder = ResNetEncoder(cfgs)
        self.classifier = RNNClassifier(cfgs)
    
    def forward(self, x):
        # x -> [B, L, N]
        B, L, N = x.size()
        state = self.classifier.init_hidden(B).cuda()
        _x = None
        for i in range(L):
            _x = x[:, i]
            #y, _x = self.encoder(_x)
            _x, state = self.classifier(_x, state)
        return _x
    
    def inference(self, x):
        time1 = time.time()
        y = torch.softmax(self.forward(x), -1)
        pred = y.argmax(dim=-1)
        time2 = time.time()
        return pred, time2 - time1

class ResNetEncoder(nn.Module):
    def __init__(self, cfgs):
        super(ResNetEncoder, self).__init__()
        self.embedding = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(.2)
        self.ln = nn.LayerNorm(1024)
        self.classifier = nn.Linear(1024, cfgs["num_classes"])
        
    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x = self.ln(x)
        y = self.classifier(x)
        return y, x
    
    def inference(self, x):
        x = x.reshape([-1, 1024])
        y, _ = self.forward(x)
        y = F.sigmoid(y)
        y = y.mean(dim=-2)
        pred = y.argmax(dim=-1)
        return pred


class RNNClassifier(nn.Module):
    def __init__(self, cfgs):
        super(RNNClassifier, self).__init__()
        self.embed_dim = cfgs["embed_dim"]

        self.fc1 = nn.Linear(1024, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.fc3 = nn.Linear(self.embed_dim, cfgs["num_classes"])

        self.activation = nn.ReLU()

    def forward(self, x, state):
        x = self.fc1(x)
        state = self.fc2(state)
        state = self.activation(state + x) # 这个地方模拟的就是RNN的结果。
        output = self.fc3(state)
        return output, state

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.embed_dim)




class MIX_RNN(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.encoder = Mix_Encoder(cfgs)
        self.classifier = MIXRNNClassifier(cfgs)
    
    def forward(self, x):
        # x -> [B, L, N]
        _, x = self.encoder(x)
        B, L, N = x.size()
        state = self.classifier.init_hidden(B).cuda()
        _x = None
        for i in range(L):
            _x = x[:, i]
            #y, _x = self.encoder(_x)
            _x, state = self.classifier(_x, state)
        return _x, _
    def inference(self, x):
        time1 = time.time()
        y1, y2 = self.forward(x)
        y = torch.softmax(y1, -1)
        pred = y.argmax(dim=-1)
        time2 = time.time()
        return pred, time2 - time1


class Mix_Encoder(nn.Module):
    def __init__(self, cfgs):
        super(Mix_Encoder, self).__init__()
        self.embedding = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(.2)
        self.ln = nn.LayerNorm(1024)
        self.classifier = nn.Linear(1024, cfgs["num_classes"])
        
    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x = self.ln(x)
        y = self.classifier(x)
        return y, x
    '''
    def inference(self, x):
        x = x.reshape([-1, 1024])
        y, _ = self.forward(x)
        y = F.sigmoid(y)
        y = y.mean(dim=-2)
        pred = y.argmax(dim=-1)
        return pred
    '''

class MIXRNNClassifier(nn.Module):
    def __init__(self, cfgs):
        super(MIXRNNClassifier, self).__init__()
        self.embed_dim = cfgs["embed_dim"]

        self.fc1 = nn.Linear(1024, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.fc3 = nn.Linear(self.embed_dim, cfgs["num_classes"])

        self.activation = nn.ReLU()

    def forward(self, x, state):
        x = self.fc1(x)
        state = self.fc2(state)
        state = self.activation(state + x) # 这个地方模拟的就是RNN的结果。
        output = self.fc3(state)
        return output, state

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.embed_dim)