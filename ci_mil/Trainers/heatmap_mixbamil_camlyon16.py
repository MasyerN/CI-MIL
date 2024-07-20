from .base import BaseTrainer
import numpy as np
import torch
from torch.nn import functional as F
import time
import h5py
import openslide
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
class Origin_ABMIL(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bag_loader = bag_loader

    def _train_epoch(self, epoch):
        self.bag_loader.dataset.set_mode('bag')
        #pred = self._inference_for_selection(self.bag_loader)
        #self.train_loader.dataset.top_k_select(pred, is_in_bag=True)
        #self.train_loader.dataset.set_mode('selected_bag')
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
            #pre_features = self.model.pre_features
            #pre_weight1 = self.model.pre_weight1
            #weight1, pre_features, pre_weight1 = reweighting.weight_learner(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
            #self.model.pre_features.data.copy_(pre_features)
            #self.model.pre_weight1.data.copy_(pre_weight1)
            output= self.model(input)

            loss1 = self.criterion(output, targets)
            #t = targets.unsqueeze(0)
            #t = targets.repeat(_.size()[1])
            #loss2 = self.criterion(_[0], targets.repeat(_.size()[1]))
            loss = loss1# + loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
            #print(f"\rTraining\t{(i+1)/len(self.train_loader)*100:.2f}%\tloss: {loss.item():.5f}", end='', flush=True)
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
                #print(f'\rinference progress: {(i+1)/len(loader)*100:.1f}%', end='', flush=True)
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
                #pre_features = self.model.pre_features.detach().clone()
                #pre_weight1 = self.model.pre_weight1.detach().clone()
                #weight1 = reweighting.weight_learner2(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
                output = self.model.inference(input)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
                probs.append(output.detach().cpu().numpy())
                targets.append(target.numpy())
                #del weight1
                # print(f'inference progress: {i+1}/{len(loader)}')
        return probs, targets

    def train(self):
        for epoch in range(self.cfgs["epochs"]): 
            self._train_epoch(epoch)
            # Validation
            self.test_loader.dataset.set_mode('bag')
            #pred = self._inference_for_selection(self.test_loader)
            #self.test_loader.dataset.top_k_select(pred, is_in_bag=True)
            #self.test_loader.dataset.set_mode('selected_bag')

            pred, target = self.inference(self.test_loader, epoch)
            score = self.metric_ftns(target, pred)
            #info = f'Epoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n'
            #print(f'Validation\tEpoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}\tprecision: {score["precision"]}\trecall: {score["recall"]}\tACC: {score["acc"]}')
            print(epoch, 'Validation:', score)
            #self._check_best(epoch, score)
            #self.logger(info)
            torch.cuda.empty_cache()
        #score = self.best_metric_info
        #print(score)
        #info = 20*'#' + f'\nf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n' + 20*'#'
        #self.logger(info)


class MixABMIL(BaseTrainer):
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
            # [1*k, N] -> [B*K, N]
            # if (i+1) % self.cfgs["batch_size"] == 0 or (i+1) == len(self.bag_loader):
            input = feature.cuda()
            targets = target.cuda()
            #maxpooling_out, _ = self.model.encoder(input.detach().clone()[0])
            #selected_num = torch.argsort(torch.softmax(maxpooling_out, 1)[:, 1], descending=True)
            #loss2 = self.criterion(torch.index_select(maxpooling_out, dim=0, index=selected_num[: 2]), targets.repeat(2))
            #pre_features = self.model.pre_features
            #pre_weight1 = self.model.pre_weight1
            #weight1, pre_features, pre_weight1 = reweighting.weight_learner(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
            #self.model.pre_features.data.copy_(pre_features)
            #self.model.pre_weight1.data.copy_(pre_weight1)
            output, _= self.model(input)
            selected_num = torch.argsort(torch.softmax(_, 1)[:, 1], descending=True)       
            loss1 = self.criterion(output, targets)
            loss2 = self.criterion(torch.index_select(_, dim=0, index=selected_num[: 2]), targets.repeat(2))
            #t = targets.unsqueeze(0)
            #t = targets.repeat(_.size()[1])
            #loss2 = self.criterion(_[0], targets.repeat(_.size()[1]))
            if epoch >= self.cfgs["asynchronous"]:
                loss = loss1 + loss2
            else:
                loss = loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
            #print(f"\rTraining\t{(i+1)/len(self.train_loader)*100:.2f}%\tloss: {loss.item():.5f}", end='', flush=True)
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
                output, _ = self.model.encoder(input[0])  # [B, num_classes]
                probs.extend(output.detach().cpu().numpy())
                #print(f'\rinference progress: {(i+1)/len(loader)*100:.1f}%', end='', flush=True)
            print("")
            probs = np.array(probs)
        return probs

    def inference(self, loader):
        self.model.eval()
        probs = []
        targets = []
        attention = []
        id_list = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                #pre_features = self.model.pre_features.detach().clone()
                #pre_weight1 = self.model.pre_weight1.detach().clone()
                #weight1 = reweighting.weight_learner2(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
                id_list.append(slide_id[0])
                output, atten = self.model.inference2(input)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
                probs.append(output.detach().cpu().numpy())
                targets.append(target.numpy())
                attention.append(atten.detach().cpu().tolist())
                #del weight1
                # print(f'inference progress: {i+1}/{len(loader)}')
        return probs, targets, attention, id_list

    def get_score(self):
        self.model.eval()
        self.test_loader.dataset.set_mode('bag')
        pred = self._inference_for_selection(self.test_loader)
        slected_idx = self.test_loader.dataset.top_k_select(pred, is_in_bag=True)
        self.test_loader.dataset.set_mode('selected_bag')
        pred, target, attention, slides_id = self.inference(self.test_loader)
        score = self.metric_ftns(target, pred)
        print(score)

        return attention, slides_id, slected_idx
    


    def draw_heatmap(self, img, points, values):
        # 读取原始图片
        #img = cv2.imread(img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 归一化values到[0,1]之间
        norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))

        # 使用matplotlib的colormap将归一化的值映射到颜色
        cmap = plt.get_cmap('hot')  # 热图的colormap
        heatmap_colors = (cmap(norm_values)[:, :3] * 255).astype(np.uint8)

        # 在图像上绘制对应的矩形区域
        for point_, color in zip(points, heatmap_colors):
            point = [point_[1] * 2, point_[0] * 2]
            top_left = tuple(map(int, point))
            bottom_right = (top_left[0] + 224, top_left[1] + 224)
            cv2.rectangle(img, top_left, bottom_right, color.tolist(), -1)

        return img



    def heatmap(self):
        self.model = torch.load('/home/omnisky/sde/NanTH/result/mix_abmil/115_0.90.pth', map_location='cuda:0')
        #self.model.load_state_dict(checkpoint)
        attention, slides_sel2, selected1 = self.get_score()
        for i in range(len(selected1)):
            for j in range(len(slides_sel2)):
                if selected1[i][0] == slides_sel2[j]:
                    h5_file = '/home/omnisky/sde/NanTH/camlyon16/testing/features/h5_files/' + selected1[i][0] + '.h5'
                    file = h5py.File(h5_file, "r")
                    coords = file['coords']
                    points = []
                    for k in selected1[i][2]:
                        points.append(coords[k])                    
                    slide = openslide.OpenSlide('/home/omnisky/sde/NanTH/camlyon16/testing/images/' + selected1[i][0] + '.tif')
                    full_img = np.array(slide.read_region((0, 0), 1, slide.level_dimensions[1]).convert('RGB'))
                    #mask = np.zeros((full_img.shape[0],full_img.shape[1]), np.uint8)
                    value = np.array(attention[j])
                    img = self.draw_heatmap(full_img, points, value)
                    cv2.imwrite('/home/omnisky/sde/NanTH/result/mix_abmil/heatmap/' + selected1[i][0] + '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY,10])
                    break



class MixABMIL_Test(BaseTrainer):
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
            # [1*k, N] -> [B*K, N]
            # if (i+1) % self.cfgs["batch_size"] == 0 or (i+1) == len(self.bag_loader):
            input = feature.cuda()
            targets = target.cuda()
            #maxpooling_out, _ = self.model.encoder(input.detach().clone()[0])
            #selected_num = torch.argsort(torch.softmax(maxpooling_out, 1)[:, 1], descending=True)
            #loss2 = self.criterion(torch.index_select(maxpooling_out, dim=0, index=selected_num[: 2]), targets.repeat(2))
            #pre_features = self.model.pre_features
            #pre_weight1 = self.model.pre_weight1
            #weight1, pre_features, pre_weight1 = reweighting.weight_learner(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
            #self.model.pre_features.data.copy_(pre_features)
            #self.model.pre_weight1.data.copy_(pre_weight1)
            output, _= self.model(input)
            selected_num = torch.argsort(torch.softmax(_, 1)[:, 1], descending=True)       
            loss1 = self.criterion(output, targets)
            loss2 = self.criterion(torch.index_select(_, dim=0, index=selected_num[: 2]), targets.repeat(2))
            #t = targets.unsqueeze(0)
            #t = targets.repeat(_.size()[1])
            #loss2 = self.criterion(_[0], targets.repeat(_.size()[1]))
            if epoch >= self.cfgs["asynchronous"]:
                loss = loss1 + loss2
            else:
                loss = loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
            #print(f"\rTraining\t{(i+1)/len(self.train_loader)*100:.2f}%\tloss: {loss.item():.5f}", end='', flush=True)
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
        self.model1.eval()
        probs = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                output, _ = self.model1.encoder(input[0])  # [B, num_classes]
                probs.extend(output.detach().cpu().numpy())
                #print(f'\rinference progress: {(i+1)/len(loader)*100:.1f}%', end='', flush=True)
            print("")
            probs = np.array(probs)
        return probs

    def inference(self, loader):
        self.model1.eval()
        probs = []
        targets = []
        attention = []
        id_list = []
        idx_list = []
        with torch.no_grad():
            for i, (feature, target, slide_id, sleceted_idx) in enumerate(loader):
                input = feature.cuda()
                #pre_features = self.model.pre_features.detach().clone()
                #pre_weight1 = self.model.pre_weight1.detach().clone()
                #weight1 = reweighting.weight_learner2(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
                id_list.append(slide_id[0])
                output, atten = self.model1.inference2(input)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
                probs.append(output.detach().cpu().numpy())
                targets.append(target.numpy())
                attention.append(atten.detach().cpu().tolist())
                idx_list.append(sleceted_idx)
                #del weight1
                # print(f'inference progress: {i+1}/{len(loader)}')
        return probs, targets, attention, id_list, idx_list

    def get_score(self):
        self.model.eval()
        self.test_loader.dataset.set_mode('bag')
        pred = self._inference_for_selection(self.test_loader)
        self.test_loader.dataset.top_k_select(pred, is_in_bag=True)
        self.test_loader.dataset.set_mode('selected_bag')
        pred, target, attention, slides_id, idx = self.inference(self.test_loader)
        #score = self.metric_ftns(target, pred)
        #print(score)

        return attention, slides_id, idx
    

    def generate(self, array):
        # Create an empty RGBA image
        heatmap = np.ones((*array.shape, 4))

        # Find the indices of the elements that are greater than 0
        indices = np.where(array > 0)
        #array[indices] = np.exp(array[indices]) / np.sum(np.sum(array))
        # Normalize the values that are greater than 0 to 0-1 range
        array[indices] = array[indices] - array[indices].min() + 0.1
        array_normalized = array[indices] / array[indices].max()

        # Define a color map from blue to red
        cmap = mcolors.LinearSegmentedColormap.from_list("", ['blue', "cyan", "yellow", "red"])

        # Apply the color map to the non-zero elements
        heatmap[indices] = cmap(array_normalized)

        # Make the white areas 50% transparent
        white_areas = np.all(heatmap == [1, 1, 1, 1], axis=-1)
        heatmap[white_areas] = [1, 1, 1, 1]

        return heatmap



    def draw_heatmap(self, img, points, values, slide):
        norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))
        heatmap = np.zeros_like(img[:, :, 0], dtype=np.float32)
        for [x, y], value in zip(points, norm_values):
            if value >= 0.8:
                heatmap[y:y+224, x:x+224] = value
        vmax = np.max(norm_values)
        vmin = np.min(norm_values)
        fig, ax = plt.subplots()
        img = cv2.resize(img, dsize=(int(img.shape[1] / 4), int(img.shape[0] / 4)), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.resize(heatmap, dsize=(int(heatmap.shape[1] / 4), int(heatmap.shape[0] / 4)), interpolation=cv2.INTER_CUBIC)
        alpha_matrix = np.zeros_like(heatmap, dtype=np.float32)
        alpha_matrix[heatmap > 0] = 0.6 
        heatmap_rgba = plt.cm.jet(heatmap)
        heatmap_rgba[..., 3] = alpha_matrix
        ax.imshow(img, cmap='gray')
        ax.imshow(heatmap_rgba, cmap='jet', vmin=vmin, vmax=vmax)
        #cax = ax.imshow(heatmap_rgba, alpha=0.6, cmap='jet', vmin=vmin, vmax=vmax)
        #cbar = fig.colorbar(cax)

        plt.savefig('/home/omnisky/sde/NanTH/result/mix_abmil/heatmap/2_' + slide + '.jpg', dpi = 3000)
        # 读取原始图片
  

        # 归一化values到[0,1]之间
        



    def heatmap(self):
        self.model1 = torch.load('/home/omnisky/sde/NanTH/result/mix_abmil/camelyon/78_0.88.pth', map_location='cuda:0')
        #self.model.load_state_dict(checkpoint)
        attention, slides, id = self.get_score()
        for i in range(len(slides)):
            h5_file = '/home/omnisky/sde/NanTH/camlyon16/testing/cp_files/patches/' + slides[i] + '.h5'
            file = h5py.File(h5_file, "r")
            coords = file['coords']
            points = []
            for k in id[i]:
            #for k in coords:
                points.append(coords[k])                    
            slide = openslide.OpenSlide('/home/omnisky/sde/NanTH/camlyon16/testing/images/' + slides[i] + '.tif')
            full_img = np.array(slide.read_region((0, 0), 1, slide.level_dimensions[1]).convert('RGB'))
            
            value = np.array(attention[i])
            values = [0 for x in value]
            sorted = np.argsort(value)
            for p in range(len(sorted)):
                values[sorted[p]] = p
            #value = np.array([1 for x in range(len(points))])
            self.draw_heatmap(full_img, points, values, slides[i])
            #plt.axis('off')
            #plt.imshow(full_img)
            #plt.imshow(heatmap, alpha=0.5)
            
            #cv2.imwrite('/home/omnisky/sde/NanTH/result/mix_abmil/heatmap/check_' + slides[i] + '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY,1])
            