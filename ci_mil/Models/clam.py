import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttnNet(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(AttnNet, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes


class AttnNetGated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(AttnNetGated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAMSB(nn.Module):
    def __init__(self, cfgs):
        super(CLAMSB, self).__init__()
        self.cfgs = cfgs
        self.size_dict = {"small": [512, 512, 256], "big": [512, 512, 384]}
        size = self.size_dict[cfgs["size_arg"]]
        attn_layer = AttnNetGated(size[1], size[2], n_classes=cfgs["num_classes"]) if cfgs["gate"] else AttnNet(size[1], size[2], n_classes=cfgs["num_classes"])

        self.attention_net = nn.Sequential(
            nn.Linear(size[0], size[1]), 
            nn.ReLU(),
            nn.Dropout(0.25),
            attn_layer
        )

        self.classifiers = nn.Linear(size[1], 1)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(cfgs["num_classes"])]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = cfgs["k"]
        self.instance_loss_fn = nn.CrossEntropyLoss()
        self.num_classes = cfgs["num_classes"]


    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
    
    @staticmethod
    def create_positive_targets(length):
        return torch.full((length, ), 1).long().cuda()
    @staticmethod
    def create_negative_targets(length):
        return torch.full((length, ), 0).long().cuda()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, attn, x, classifier): 
        k = self.k_sample if self.k_sample < x.size(0) else x.size(0)
        top_p_ids = torch.topk(attn, k)[1]
        top_p = torch.index_select(x, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-attn, k)[1]
        top_n = torch.index_select(x, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k)
        n_targets = self.create_negative_targets(k)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, attn, x, classifier):
        k = self.k_sample if self.k_sample < x.size(0) else x.size(0)
        top_p_ids = torch.topk(attn, k)[1]
        top_p = torch.index_select(x, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(k)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, x, target=None, instance_eval=False, return_features=False, attention_only=False):
        x = x.reshape([-1, 512])           # L x N
        attn, x = self.attention_net(x)     # L x num_classes
        attn = torch.transpose(attn, 1, 0)  # num_classes x L
        if attention_only:
            return attn
        attn_raw = attn
        attn = F.softmax(attn, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(target, num_classes=self.num_classes).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                _attn = attn[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(_attn, x, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    instance_loss, preds, targets = self.inst_eval_out(_attn, x, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                total_inst_loss += instance_loss

            total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(attn, x)  # (num_classes x L) * (L x N) -> num_classes x N
        logits = self.classifiers(M).T  # num_classes x 1
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, attn_raw, results_dict
    
    # def inference(sample)

class CLAMMB(CLAMSB):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        nn.Module.__init__(self)
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = AttnNetGated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = AttnNet(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) 
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
