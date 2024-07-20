import torch
from numpy import inf
import numpy as np
import os, time 
import os.path as osp


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, cfgs):
        self.cfgs = cfgs

        # setup GPU device if available, move model into configured device
        self.model = model.cuda()
        # self.model = torch.nn.DataParallel(model)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.cfgs["epochs"]
        self.save_period = self.cfgs["save_period"]

        self.mnt_mode = self.cfgs["monitor_mode"]
        self.mnt_metric = self.cfgs["monitor_metric"]
        assert self.mnt_mode in ['min', 'max']
        self.best_metric_info = None

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf

        if osp.exists(osp.join("outputs", self.cfgs["task"])) is False:
            os.mkdir(osp.join("outputs", self.cfgs["task"]))
        self.checkpoint_path = osp.join("outputs", self.cfgs["task"])
        self.logger_path = osp.join("outputs", self.cfgs["task"], "resluts.log")
        with open(self.logger_path, 'w') as f:
            f.write(f'{time.ctime()}\n{self.cfgs}\n')

    def train(self):
        pass

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def inference(self):
        pass
    
    def _check_best(self, epoch, metric):
        is_best = False
        if self.mnt_mode == 'max' and metric[self.mnt_metric] > self.mnt_best:
            is_best = True
            self.mnt_best = metric[self.mnt_metric]
            self.best_metric_info = metric
        elif self.mnt_mode == 'min' and metric[self.mnt_metric] < self.mnt_best:
            is_best = True    
            self.mnt_best = metric[self.mnt_metric]
            self.best_metric_info = metric
        else:
            is_best = False
        self._save_checkpoint(epoch, is_best)
        return is_best

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'seed': self.cfgs['seed']
        }
        filename = os.path.join(self.checkpoint_path, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_path, 'model_best.pth')
            torch.save(state, best_path)
            print("*************** Saving current best: model_best.pth ... ***************")
    
    def logger(self, info):
        with open(self.logger_path, 'a') as f:
            f.write(info)
