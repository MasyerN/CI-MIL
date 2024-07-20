from .base import BaseTrainer
import numpy as np
import torch
from torch.nn import functional as F
import time


class Supvision(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def _train_epoch(self, epoch):
        
        loss = self._train_iter()

        # logger
        print(f'Training\tEpoch: [{epoch+1}/{self.cfgs["epochs"]}]\tLoss: {loss}')
    
    def _train_iter(self):
        self.model.train()
        running_loss = 0.
        for i, (feature, target, slide_id) in enumerate(self.train_loader):
            # [1*k, N] -> [B*K, N]
            input = feature.cuda()
            target = target.cuda()
            output = self.model(input)
            target = target
            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(1)
        return running_loss/len(self.train_loader.dataset)
    
    def _inference_for_selection(self, loader):
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                output = self.model(input.reshape([-1, 1024]))  # [B, num_classes]
                probs.extend(output.detach().cpu().numpy())
                print(f'\rinference progress: {(i+1)/len(loader)*100:.1f}%', end='', flush=True)
            print("")
            probs = np.array(probs).reshape([-1, self.cfgs["num_classes"]])
        return probs

    def inference(self, loader, k=5):
        self.model.eval()
        probs = []
        preds = []
        running_loss = 0.
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                pred, prob, out = self.model.inference(input[0])
                target = target.cuda().repeat(out.size()[0])
                loss = self.criterion(out, target)
                probs.append(prob.detach().cpu().numpy())
                preds.append(pred.detach().cpu().tolist())
                running_loss += loss.item()*input.size(1)
                # print(f'inference progress: {i+1}/{len(loader)}')
        return preds, probs, running_loss/len(loader.dataset)
 
    def train(self):
        #l oop throuh epochs
        chose_slide = [[] for i in range(self.cfgs["epochs"])]
        self.train_loader.dataset.set_mode('instance')
        self.test_loader.dataset.set_mode('bag')
        for epoch in range(self.cfgs["epochs"]): 
            self._train_epoch(epoch)
            # Validation            
            pred, prob, loss = self.inference(self.test_loader)
            score = self.metric_ftns(np.array(self.test_loader.dataset.targets).reshape([-1, 1]), pred, np.array(prob))
            for test_index in range(len(self.test_loader.dataset.targets)):
                if self.test_loader.dataset.targets[test_index] == pred[test_index]:
                    chose_slide[epoch].append(self.test_loader.dataset.data_info[test_index]["slide_id"])
            #print(len(chose_slide[epoch]))
            #info = f'Epoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n'
            #print(f'Validation\tEpoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}\tprecision: {score["precision"]}\trecall: {score["recall"]}\tACC: {score["acc"]}')
            print('tetsloss:', loss)
            print(score)
            print('---------------------------------------------------')
            '''
            print(epoch, result)
            print('===============================================')
            '''
            #self._check_best(epoch, score)
            #self.logger(info)
        # torch.save(chose_slide, r'/home/omnisky/verybigdisk/NanTH/Result/meanmil_inf.pth')
        # score = self.best_metric_info
        # info = 20*'#' + f'\nf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n' + 20*'#'
        # print(print(f'Validation_best\tf1: {score["f1"]}\tprecision: {score["precision"]}\trecall: {score["recall"]}\tACC: {score["acc"]}'))
        # self.logger(info)

