from .base import BaseTrainer
import numpy as np
import torch
from torch.nn import functional as F
import time
import config
from config import parser as reweight_parser
stable_cfg = reweight_parser.parse_args()
#import reweighting.weight_learner2 as weight_learner
import reweighting
class ABMILTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bag_loader = bag_loader

    def _train_epoch(self, epoch):
        self.bag_loader.dataset.set_mode('bag')
        pred = self._inference_for_selection(self.bag_loader)
        self.train_loader.dataset.top_k_select(pred, is_in_bag=True)
        self.train_loader.dataset.set_mode('selected_bag')
        loss = self._train_iter(epoch)
        
        # logger
        print(f'Training\tEpoch: [{epoch+1}/{self.cfgs["epochs"]}]\tLoss: {loss}')
    
    def _train_iter(self, epoch):
        self.model.train()
        self.model.encoder.eval()
        running_loss = 0.
        # default batch size is 1, but fail to train. So I random select the patches in the bag level.
        # features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
        # targets = torch.zeros([self.cfgs["batch_size"]]).long()
        for i, (feature, target, slide_id) in enumerate(self.train_loader):
            # [1*k, N] -> [B*K, N]
            # if (i+1) % self.cfgs["batch_size"] == 0 or (i+1) == len(self.bag_loader):
            input = feature.cuda()
            targets = target.cuda()
            pre_features = self.model.pre_features
            pre_weight1 = self.model.pre_weight1
            weight1, pre_features, pre_weight1 = reweighting.weight_learner(input.reshape([-1, pre_features.size()[-1]]), pre_features, pre_weight1, stable_cfg, epoch, i)
            weight1 = weight1.reshape([input.size()[0], input.size()[1], 1])
            self.model.pre_features.data.copy_(pre_features)
            self.model.pre_weight1.data.copy_(pre_weight1)
            output, attn = self.model(input, weight1)

            loss = self.criterion(output, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
            print(f"\rTraining\t{(i+1)/len(self.train_loader)*100:.2f}%\tloss: {loss.item():.5f}", end='', flush=True)
            features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
            targets = torch.zeros([self.cfgs["batch_size"]]).long()
            # else:
            #     length = feature.size(1)
            #     _index = np.arange(0, length)
            #     np.random.shuffle(_index)
            #     if length > self.cfgs["sample_size"]:
            #         features[i % self.cfgs["batch_size"]] = feature[0, _index[:self.cfgs["sample_size"]]]
            #     else:
            #         features[i % self.cfgs["batch_size"], _index[:length]] = feature[0, _index[:length]]
            #     targets[i % self.cfgs["batch_size"]] = target
        print('')
        return running_loss/len(self.train_loader.dataset)

    def _inference_for_selection(self, loader):
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                output, _ = self.model.encoder(input.reshape([-1, 1024]))  # [B, num_classes]
                probs.extend(output.detach().cpu().numpy())
                print(f'\rinference progress: {(i+1)/len(loader)*100:.1f}%', end='', flush=True)
            print("")
            probs = np.array(probs).reshape([-1, self.cfgs["num_classes"]])
        return probs

    def inference(self, loader, epoch = 0):
        self.model.eval()
        probs = []
        targets = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                pre_features = self.model.pre_features.detach().clone()
                pre_weight1 = self.model.pre_weight1.detach().clone()
                weight1 = reweighting.weight_learner2(input.reshape([-1, pre_features.size()[-1]]), pre_features, pre_weight1, stable_cfg, epoch, i)
                weight1 = weight1.reshape([input.size()[0], input.size()[1], 1])
                output = self.model.inference(input, weight1)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
                probs.append(output.detach().cpu().numpy())
                targets.append(target.numpy())
                del weight1
                # print(f'inference progress: {i+1}/{len(loader)}')
        return probs, targets

    def train(self):
        for epoch in range(self.cfgs["epochs"]): 
            self._train_epoch(epoch)
            # Validation
            self.test_loader.dataset.set_mode('bag')
            pred = self._inference_for_selection(self.test_loader)
            self.test_loader.dataset.top_k_select(pred, is_in_bag=True)
            self.test_loader.dataset.set_mode('selected_bag')

            pred, target = self.inference(self.test_loader, epoch)
            score = self.metric_ftns(target, pred)
            info = f'Epoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n'
            print(f'Validation\tEpoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}\tprecision: {score["precision"]}\trecall: {score["recall"]}\tACC: {score["acc"]}')
            self._check_best(epoch, score)
            self.logger(info)
            torch.cuda.empty_cache()
        score = self.best_metric_info
        info = 20*'#' + f'\nf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n' + 20*'#'
        self.logger(info)

