from .base import BaseTrainer
import numpy as np
import torch
from torch.nn import functional as F
import time


class MILRNN(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bag_loader = bag_loader


    
    def _rnn_train_epoch(self, epoch):
        loss = self._rnn_train_iter()
        # logger
    
    
    def _rnn_train_iter(self):
        self.model.train()
        running_loss = 0.
        for i, (feature, target, slide_id) in enumerate(self.train_loader):
            # [B, k, N] -> [B, K, N]
            inputs = feature.cuda()
            target = target.cuda()
            B = inputs.size(0)
            output1, output2 = self.model(inputs)
            __ = torch.softmax(output2, -1)[:, :, 1]
            selected_num = torch.argsort(__, dim = 1, descending=True) 
            loss2 = self.criterion(torch.index_select(output2[0], dim = 0, index=selected_num[0][: 2]), target[0].repeat(2))
            for k in range(1, output2.size()[0]):
                loss2 += self.criterion(torch.index_select(output2[k], dim=0, index=selected_num[k][: 2]), target[k].repeat(2))
            loss2 = loss2 / output2.size()[0]
            loss1 = self.criterion(output1, target)
            loss = loss1 + loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*inputs.size(0)
        return running_loss/len(self.train_loader.dataset)
    
    def _inference_for_selection(self, loader):
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                output, _ = self.model.encoder(input[0])  # [B, num_classes]
                probs.extend(output.detach().cpu().numpy())
            print("")
            probs = np.array(probs).reshape([-1, self.cfgs["num_classes"]])
        return probs
    '''
    def inference(self, loader, k=5):
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                output = self.model.encoder.inference(input)
                probs.append(output.detach().cpu().numpy())
                # print(f'inference progress: {i+1}/{len(loader)}')
        return probs
    '''
    def rnn_inference(self, loader, k=5):
        self.model.eval()
        probs = []
        targets = []
        time_all = 0
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                inputs = feature.cuda()
                output, time = self.model.inference(inputs)
                time_all += time
                probs.append(output.detach().cpu().numpy())
                targets.append(target.numpy())
        return probs, targets, time_all

    def train(self):
        # ----------------
        # Train MIL model
        # ----------------
        """
        model = self.model
        for epoch in range(self.cfgs["epochs"]): 
            self._train_epoch(epoch)
            # Validation
            self.test_loader.dataset.set_mode('bag')
            pred = self.inference(self.test_loader)
            score = self.metric_ftns(self.test_loader.dataset.targets, pred)
            info = f'*MIL STAGE* Epoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n'
            print(f'*MIL STAGE* Validation\tEpoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}\tprecision: {score["precision"]}\trecall: {score["recall"]}\tACC: {score["acc"]}')
            is_best = self._check_best(epoch, score)
            if is_best:
                model = self.model
            self.logger(info)
        score = self.best_metric_info
        info = 20*'#' + f'\nf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n' + 20*'#'
        self.logger(info)
        self.model = model
        """
        checkpoint = torch.load("outputs/mean_pooling_mil/model_best.pth")
        self.model.encoder.load_state_dict(checkpoint['state_dict'])
        # ----------------
        # Train RNN model
        # ----------------
        

        self.test_loader.dataset.set_mode('bag')
        pred = self._inference_for_selection(self.test_loader)
        self.test_loader.dataset.top_k_select(pred, is_in_bag=True)
        self.test_loader.dataset.set_mode('selected_bag')
        chose_slide = [[] for i in range(self.cfgs["epochs"])]
        for epoch in range(self.cfgs["epochs"]): 
            self.bag_loader.dataset.set_mode('bag')
            pred = self._inference_for_selection(self.bag_loader)
            self.train_loader.dataset.top_k_select(pred, is_in_bag=True)
            self.train_loader.dataset.set_mode('selected_bag')
            self._rnn_train_epoch(epoch)
            self.test_loader.dataset.set_mode('bag')
            pred = self._inference_for_selection(self.test_loader)
            self.test_loader.dataset.top_k_select(pred, is_in_bag=True)
            self.test_loader.dataset.set_mode('selected_bag')
            # Validation
            pred, target, time_all = self.rnn_inference(self.test_loader)
            score = self.metric_ftns(target, pred)
            for test_index in range(len(self.test_loader.dataset.targets)):
                if self.test_loader.dataset.targets[test_index] == pred[test_index]:
                    chose_slide[epoch].append(self.test_loader.dataset.data_info[test_index]["slide_id"])
            #print(len(chose_slide[epoch]))
            print(epoch, score)
            print('==================================================================')

