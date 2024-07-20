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

class CLAMTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bag_loader = bag_loader

    def _train_epoch(self, epoch):
        self.train_loader.dataset.set_mode('bag')
        loss = self._train_iter()

        # logger
        print(f'Training\tEpoch: [{epoch+1}/{self.cfgs["epochs"]}]\tLoss: {loss}')
    
    def _train_iter(self):
        self.model.train()
        running_loss = 0.
        inst_count = 0
        for i, (feature, target, slide_id) in enumerate(self.train_loader):
            # [1*k, N] -> [B*K, N]
            input = feature.cuda()
            target = target.cuda()
            output, _, _, _, instance_dict = self.model(input, target, instance_eval=True)

            loss_bag = self.criterion(output, target)
            instance_loss = instance_dict['instance_loss']
        
            loss = self.cfgs["bag_weight"] * loss_bag + (1-self.cfgs["bag_weight"]) * instance_loss 

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
        return running_loss/len(self.train_loader.dataset)
    
    def inference(self, loader, k=5):
        self.model.eval()
        probs = []
        preds = []
        targets = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                logits, Y_prob, Y_hat, attn_raw, results_dict = self.model(input)
                preds.append(Y_hat[0].detach().cpu().numpy())
                probs.append(Y_prob.detach().cpu().tolist()[0])
                targets.append(target.tolist())
                # print(f'inference progress: {i+1}/{len(loader)}')
        return preds, np.array(probs), np.array(targets)

    def train(self):
        save_log = []
        for epoch in range(self.cfgs["epochs"]): 
            self._train_epoch(epoch)
            # Validation
            self.test_loader.dataset.set_mode('bag')
            pred, prob, target = self.inference(self.test_loader)
            save_log.append({'t': target, 'p': pred})
            draw_confusion_matrix(target, pred, 'CLAM_F_30_' + str(epoch))
            score = self.metric_ftns(target, pred, prob)
            print(epoch, score)
        torch.save(save_log, r'/home/omnisky/sde/NanTH/result/confusion_matrix/clam_f_30.pth')

