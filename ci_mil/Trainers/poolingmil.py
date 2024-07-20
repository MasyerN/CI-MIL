from .base import BaseTrainer
import numpy as np
import torch
from torch.nn import functional as F
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def draw_confusion_matrix(true_labels, predicted_labels, name):
    fontsize = 14

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # 设置类别名称
    class_names = ['N', 'B', 'M', 'SCC', 'SK']

    # 绘制混淆矩阵图
    plt.figure(figsize=(4, 3))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    #plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=fontsize)
    plt.yticks(tick_marks, class_names, fontsize=fontsize)

    # 添加标签
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(conf_matrix[i, j]), fontsize=11, horizontalalignment='center', color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')

    #plt.ylabel('Target')
    #plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()
    plt.savefig('/home/omnisky/sde/NanTH/result/confusion_matrix/' + name + '.png')

class PoolingMILTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bag_loader = bag_loader

    def _train_epoch(self, epoch):
        self.bag_loader.dataset.set_mode('bag')
        pred = self._inference_for_selection(self.bag_loader)
        self.train_loader.dataset.top_k_select(pred)
        self.train_loader.dataset.set_mode('selected_instance')
        loss = self._train_iter()

        # logger
        #print(f'Training\tEpoch: [{epoch+1}/{self.cfgs["epochs"]}]\tLoss: {loss}')
    
    def _train_iter(self):
        self.model.train()
        running_loss = 0.
        for i, (feature, target, slide_id) in enumerate(self.train_loader):
            # [1*k, N] -> [B*K, N]
            input = feature.cuda()
            target = target.cuda()
            output = self.model(input)

            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
        return running_loss/len(self.train_loader.dataset)
    
    def _inference_for_selection(self, loader):
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                output = self.model(input.reshape([-1, 1024]))  # [B, num_classes]
                probs.extend(torch.softmax(output, -1).detach().cpu().numpy())
                #print(f'\rinference progress: {(i+1)/len(loader)*100:.1f}%', end='', flush=True)
            print("")
            probs = np.array(probs).reshape([-1, self.cfgs["num_classes"]])
        return probs

    def inference(self, loader, k=5):
        self.model.eval()
        probs = []
        preds = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                pred, prob = self.model.inference(input)
                probs.append(prob.detach().cpu().numpy())
                preds.append(pred.detach().cpu().tolist())
                # print(f'inference progress: {i+1}/{len(loader)}')
        return preds, probs
 
    def train(self):
        #l oop throuh epochs
        save_log = []
        chose_slide = [[] for i in range(self.cfgs["epochs"])]
        for epoch in range(self.cfgs["epochs"]): 
            self._train_epoch(epoch)
            # Validation
            self.test_loader.dataset.set_mode('bag')
            pred, prob = self.inference(self.test_loader)
            draw_confusion_matrix(np.array(self.test_loader.dataset.targets).reshape([-1, 1]), pred, 'MAX_L_Epr_' + str(epoch))
            save_log.append({'t': np.array(self.test_loader.dataset.targets).reshape([-1, 1]), 'p': pred})
            score = self.metric_ftns(np.array(self.test_loader.dataset.targets).reshape([-1, 1]), pred, np.array(prob))
            for test_index in range(len(self.test_loader.dataset.targets)):
                if self.test_loader.dataset.targets[test_index] == pred[test_index]:
                    chose_slide[epoch].append(self.test_loader.dataset.data_info[test_index]["slide_id"])
            #print(len(chose_slide[epoch]))
            #info = f'Epoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n'
            #print(f'Validation\tEpoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}\tprecision: {score["precision"]}\trecall: {score["recall"]}\tACC: {score["acc"]}')
            print(score)
            print('---------------------------------------------------')
            '''
            print(epoch, result)
            print('===============================================')
            '''
        torch.save(save_log, r'/home/omnisky/sde/NanTH/result/confusion_matrix/max_l_epr.pth')
            #self._check_best(epoch, score)
            #self.logger(info)
        # torch.save(chose_slide, r'/home/omnisky/verybigdisk/NanTH/Result/meanmil_inf.pth')
        # score = self.best_metric_info
        # info = 20*'#' + f'\nf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n' + 20*'#'
        # print(print(f'Validation_best\tf1: {score["f1"]}\tprecision: {score["precision"]}\trecall: {score["recall"]}\tACC: {score["acc"]}'))
        # self.logger(info)

