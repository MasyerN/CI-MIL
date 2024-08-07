import torch
import torch.nn as nn
import torch.nn.functional as F



class ABMIL(nn.Module):
    def __init__(self, cfgs):
        super(ABMIL, self).__init__()
        self.L = cfgs['feature_dim']
        self.D = int(cfgs['feature_dim'] / 2)
        self.K = 1

        #self.encoder = MeanPoolingMIL(cfgs)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Dropout(.2),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, cfgs["num_classes"]),
            # nn.Softmax(dim=-1)
            # nn.Sigmoid()
        )
    def forward(self, x):

        attn = self.attention(x).softmax(dim=1)
        m = torch.matmul(attn.transpose(-1, -2), x).squeeze(1)  # Bx1xN
        y = self.classifier(m)
        # y_hat = torch.ge(y_prob, 0.5).float()
        return y, attn

    # AUXILIARY METHODS
    def inference(self, x):
        y, _ = self.forward(x)
        return y.softmax(dim=1), y.argmax(dim=-1)

 
class MixABMIL(nn.Module):
    def __init__(self, cfgs):
        super(MixABMIL, self).__init__()
        self.L = cfgs['feature_dim']
        self.D = int(cfgs['feature_dim'] / 2)
        self.K = 1

        self.encoder = MeanPoolingMIL(cfgs)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Dropout(.2),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, cfgs["num_classes"]),
            # nn.Softmax(dim=-1)
            # nn.Sigmoid()
        )
    def forward(self, x):
        y_, _ = self.encoder(x[0])
        attn = self.attention(_).softmax(dim=0)
        m = torch.matmul(attn.transpose(-1, -2), _) # Bx1xN
        y = self.classifier(m)
        # y_hat = torch.ge(y_prob, 0.5).float()
        return y, y_

    # AUXILIARY METHODS
    def inference(self, x):
        y, _ = self.forward(x)
        y = y.argmax(dim=-1)
        return y
    
    def inference2(self, x):
        _, x = self.encoder(x[0])
        attn = self.attention(x.unsqueeze(0)).softmax(dim=1)
        m = torch.matmul(attn.transpose(-1, -2), x).squeeze(1)  # Bx1xN
        y = self.classifier(m)
        y = y.argmax(dim=-1)
        return y, attn.transpose(-1, -2).squeeze(0).squeeze(0)


class Stable_ABMIL(nn.Module):
    def __init__(self, cfgs):
        super(Stable_ABMIL, self).__init__()
        self.L = cfgs['feature_dim']
        self.D = int(cfgs['feature_dim'] / 2)
        self.K = 1

        self.encoder = MeanPoolingMIL(cfgs)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Dropout(.2),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, cfgs["num_classes"]),
            # nn.Softmax(dim=-1)
            # nn.Sigmoid()
        )
        self.register_buffer('pre_features', torch.zeros(cfgs["k"], cfgs['feature_dim']))
        self.register_buffer('pre_weight1', torch.ones(cfgs["k"], 1))
    def forward(self, x, weight):

        _, x = self.encoder(x[0])  # BxLxN

        attn = self.attention(x.unsqueeze(0)).softmax(dim=1) # BxLx1
        attn = torch.mul(attn, weight)
        m = torch.matmul(attn.transpose(-1, -2), x).squeeze(1)  # Bx1xN
        y = self.classifier(m)
        # y_hat = torch.ge(y_prob, 0.5).float()
        return y, _

    # AUXILIARY METHODS
    def inference(self, x, weight):
        y, _ = self.forward(x, weight)
        #y = y.argmax(dim=-1)
        return y
    """
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        print(Y_prob,A)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
    """

class GatedAttention(nn.Module):
    def __init__(self, cfgs):
        super(GatedAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        self.feature_extractor_part = nn.Sequential(
            nn.Linear(1024, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, cfgs["num_classes"]),
            nn.Sigmoid()
            # nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = x.view(-1, 1024)
        x = self.feature_extractor_part(x)  # NxL

        attn_V = self.attention_V(x)  # NxD
        attn_U = self.attention_U(x)  # NxD
        attn = self.attention_weights(attn_V * attn_U) # element wise multiplication # NxK
        attn = torch.transpose(attn, 1, 0)  # KxN
        attn = F.softmax(attn, dim=1)  # softmax over N

        m = torch.mm(attn, x)  # KxL

        y = self.classifier(m)

        return y, attn

    # AUXILIARY METHODS
    def inference(self, x):
        y, _ = self.forward(x)
        y = y.argmax(dim=-1)
        return y

class MeanPoolingMIL(nn.Module): 
    def __init__(self, cfgs):
        super().__init__()
        self.embedding = nn.Linear(cfgs['feature_dim'], cfgs['feature_dim'])
        self.dropout = nn.Dropout(.2)
        self.ln = nn.LayerNorm(cfgs['feature_dim'])
        self.classifier = nn.Linear(cfgs['feature_dim'], cfgs["num_classes"])
        self.cfg = cfgs
        #self.attentionnet = ABMIL(cfgs)
        
    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x = self.ln(x)
        y = self.classifier(x)
        # y = F.sigmoid(y)
        return y, x
    
    
    
    
class IRL_ABMIL(nn.Module):
    def __init__(self, cfgs):
        super(IRL_ABMIL, self).__init__()
        self.L = cfgs['feature_dim']
        self.D = int(cfgs['feature_dim'] / 2)
        self.K = 1

        self.encoder = MeanPoolingMIL(cfgs)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Dropout(.2),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, cfgs["num_classes"]),
            # nn.Softmax(dim=-1)
            # nn.Sigmoid()
        )
    def forward(self, x):
        _, x = self.encoder(x[0])
        attn = self.attention(x.unsqueeze(0)).softmax(dim=1)
        m = torch.matmul(attn.transpose(-1, -2), x).squeeze(1)  # Bx1xN
        y = self.classifier(m)
        # y_hat = torch.ge(y_prob, 0.5).float()
        return y, _

    # AUXILIARY METHODS
    def inference(self, x):
        y, _ = self.forward(x)
        y = y.softmax(dim=-1)
        pred = y.argmax(dim=-1)
        return pred, y
    
    def inference2(self, x):
        _, x = self.encoder(x[0])
        attn = self.attention(x.unsqueeze(0)).softmax(dim=1)
        m = torch.matmul(attn.transpose(-1, -2), x).squeeze(1)  # Bx1xN
        y = self.classifier(m)
        y = y.argmax(dim=-1)
        return y, attn.transpose(-1, -2).squeeze(0).squeeze(0)