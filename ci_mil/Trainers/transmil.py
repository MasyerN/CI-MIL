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
class TransMILTrainer(BaseTrainer):
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
        self.model.encoder.train()
        running_loss = 0.
        # default batch size is 1, but fail to train. So I random select the patches in the bag level.
        # features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
        # targets = torch.zeros([self.cfgs["batch_size"]]).long()
        
        for i, (feature, target, slide_id) in enumerate(self.train_loader):
            input = feature.cuda()
            targets = target.cuda()
            pre_features = self.model.pre_features
            pre_weight1 = self.model.pre_weight1
            cfeature, _ = self.model.encoder(input)
            weight1, pre_features, pre_weight1 = reweighting.weight_learner(cfeature[0].detach().clone(), pre_features, pre_weight1, stable_cfg, epoch, i)
            self.model.pre_features.data.copy_(pre_features)
            self.model.pre_weight1.data.copy_(pre_weight1)
            output= self.model(cfeature, weight1)

            loss1 = self.criterion(output, targets)
            loss2 = self.criterion(_[0], targets.repeat(_.size()[1]))
            loss = loss1 + loss2
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
        '''
        for i, (feature, target, slide_id) in enumerate(self.train_loader):
            input = feature.cuda()
            targets = target.cuda()
            pre_features = self.model.pre_features
            pre_weight1 = self.model.pre_weight1
            weight1, pre_features, pre_weight1 = reweighting.weight_learner(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
            self.model.pre_features.data.copy_(pre_features)
            self.model.pre_weight1.data.copy_(pre_weight1)
            output, _ = self.model(input, weight1)

            loss = self.criterion(output, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
            print(f"\rTraining\t{(i+1)/len(self.train_loader)*100:.2f}%\tloss: {loss.item():.5f}", end='', flush=True)
            features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
            targets = torch.zeros([self.cfgs["batch_size"]]).long()
        '''
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
                #print(f'\rinference progress: {(i+1)/len(loader)*100:.1f}%', end='', flush=True)
            #print("")
            probs = np.array(probs).reshape([-1, self.cfgs["num_classes"]])
        return probs

    def inference(self, loader, epoch = 0):
        self.model.eval()
        self.model.encoder.eval()
        probs = []
        targets = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                cfeature, _ = self.model.encoder(input)
                pre_features = self.model.pre_features.detach().clone()
                pre_weight1 = self.model.pre_weight1.detach().clone()
                weight1 = reweighting.weight_learner2(cfeature[0].detach().clone(), pre_features, pre_weight1, stable_cfg, epoch, i)
                output = self.model.inference(cfeature, weight1)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
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
            print(epoch, score)
            print('--------------------------------------')
            #info = f'Epoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n'
            #print(f'Validation\tEpoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}\tprecision: {score["precision"]}\trecall: {score["recall"]}\tACC: {score["acc"]}')
            #self._check_best(epoch, score)
            #self.logger(info)
            torch.cuda.empty_cache()
        score = self.best_metric_info
        print(score)
        #info = 20*'#' + f'\nf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n' + 20*'#'
        #self.logger(info)





class BaseTransMILTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bag_loader = bag_loader

    def _train_epoch(self, epoch):
        self.bag_loader.dataset.set_mode('bag')
        loss = self._train_iter(epoch)
        
        # logger
        print(f'Training\tEpoch: [{epoch+1}/{self.cfgs["epochs"]}]\tLoss: {loss}')
    
    def _train_iter(self, epoch):
        self.model.train()
        #self.model.encoder.eval()
        running_loss = 0.
        # default batch size is 1, but fail to train. So I random select the patches in the bag level.
        # features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
        # targets = torch.zeros([self.cfgs["batch_size"]]).long()
        for i, (feature, target, slide_id) in enumerate(self.train_loader):
            # [1*k, N] -> [B*K, N]
            # if (i+1) % self.cfgs["batch_size"] == 0 or (i+1) == len(self.bag_loader):
            input = feature.cuda()
            targets = target.cuda()
            output = self.model(input)

            loss1 = self.criterion(output, targets)
            #t = targets.unsqueeze(0)
            #t = targets.repeat(_.size()[1])
            #loss2 = self.criterion(_[0], targets.repeat(_.size()[1]))
            loss = loss1# + loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
            print(f"\rTraining\t{(i+1)/len(self.train_loader)*100:.2f}%\tloss: {loss.item():.5f}", end='', flush=True)
            #features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
            #targets = torch.zeros([self.cfgs["batch_size"]]).long()
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
        save = []
        probs = []
        targets = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                output = self.model.inference(input)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
                probs.append(output.detach().cpu().numpy())
                targets.append(target.numpy())
                if output.detach().cpu().numpy() == target.numpy():
                    save.append(slide_id)
                # print(f'inference progress: {i+1}/{len(loader)}')
        return probs, targets, save

    def train(self):
        save_slide = []
        for epoch in range(self.cfgs["epochs"]): 
            self._train_epoch(epoch)
            # Validation
            self.test_loader.dataset.set_mode('bag')

            pred, target, slide_true = self.inference(self.test_loader, epoch)
            save_slide.append(slide_true)
            score = self.metric_ftns(target, pred)
            print(score)
            print('================================================')
            #info = f'Epoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n'
            #print(f'Validation\tEpoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}\tprecision: {score["precision"]}\trecall: {score["recall"]}\tACC: {score["acc"]}')
            #self._check_best(epoch, score)
            #self.logger(info)
            torch.cuda.empty_cache()
            #torch.save(save_slide, r'/home/omnisky/verybigdisk/NanTH/Result/transmil_inf.pth')
        #score = self.best_metric_info
        #info = 20*'#' + f'\nf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n' + 20*'#'
        #self.logger(info)



class MixTransMILTrainer(BaseTrainer):
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
        self.model.encoder.train()
        running_loss = 0.
        # default batch size is 1, but fail to train. So I random select the patches in the bag level.
        # features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
        # targets = torch.zeros([self.cfgs["batch_size"]]).long()
        
        for i, (feature, target, slide_id) in enumerate(self.train_loader):
            input = feature.cuda()
            targets = target.cuda()
            cfeature, _ = self.model.encoder(input)
            output= self.model(cfeature)

            loss1 = self.criterion(output, targets)
            loss2 = self.criterion(_[0][: 10], targets.repeat(_.size()[1])[: 10])
            loss = loss1 + loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
            print(f"\rTraining\t{(i+1)/len(self.train_loader)*100:.2f}%\tloss: {loss.item():.5f}", end='', flush=True)
            #features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
            #targets = torch.zeros([self.cfgs["batch_size"]]).long()
            # else:
            #     length = feature.size(1)
            #     _index = np.arange(0, length)
            #     np.random.shuffle(_index)
            #     if length > self.cfgs["sample_size"]:
            #         features[i % self.cfgs["batch_size"]] = feature[0, _index[:self.cfgs["sample_size"]]]
            #     else:
            #         features[i % self.cfgs["batch_size"], _index[:length]] = feature[0, _index[:length]]
            #     targets[i % self.cfgs["batch_size"]] = target
        '''
        for i, (feature, target, slide_id) in enumerate(self.train_loader):
            input = feature.cuda()
            targets = target.cuda()
            pre_features = self.model.pre_features
            pre_weight1 = self.model.pre_weight1
            weight1, pre_features, pre_weight1 = reweighting.weight_learner(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
            self.model.pre_features.data.copy_(pre_features)
            self.model.pre_weight1.data.copy_(pre_weight1)
            output, _ = self.model(input, weight1)

            loss = self.criterion(output, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
            print(f"\rTraining\t{(i+1)/len(self.train_loader)*100:.2f}%\tloss: {loss.item():.5f}", end='', flush=True)
            features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
            targets = torch.zeros([self.cfgs["batch_size"]]).long()
        '''
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
                #print(f'\rinference progress: {(i+1)/len(loader)*100:.1f}%', end='', flush=True)
            #print("")
            probs = np.array(probs).reshape([-1, self.cfgs["num_classes"]])
        return probs

    def inference(self, loader, epoch = 0):
        self.model.eval()
        self.model.encoder.eval()
        probs = []
        targets = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                cfeature, _ = self.model.encoder(input)
                #weight1 = reweighting.weight_learner2(cfeature[0].detach().clone(), pre_features, pre_weight1, stable_cfg, epoch, i)
                output = self.model.inference(cfeature)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
                probs.append(output.detach().cpu().numpy())
                targets.append(target.numpy())
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
            print(epoch, score)
            print('--------------------------------------')
            #info = f'Epoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n'
            #print(f'Validation\tEpoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}\tprecision: {score["precision"]}\trecall: {score["recall"]}\tACC: {score["acc"]}')
            #self._check_best(epoch, score)
            #self.logger(info)
            torch.cuda.empty_cache()
        score = self.best_metric_info
        print(score)
        #info = 20*'#' + f'\nf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n' + 20*'#'
        #self.logger(info)