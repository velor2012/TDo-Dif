'''
                       ::
                      :;J7, :,                        ::;7:
                      ,ivYi, ,                       ;LLLFS:
                      :iv7Yi                       :7ri;j5PL
                     ,:ivYLvr                    ,ivrrirrY2X,
                     :;r@Wwz.7r:                :ivu@kexianli.
                    :iL7::,:::iiirii:ii;::::,,irvF7rvvLujL7ur
                   ri::,:,::i:iiiiiii:i:irrv177JX7rYXqZEkvv17
                ;i:, , ::::iirrririi:i:::iiir2XXvii;L8OGJr71i
              :,, ,,:   ,::ir@mingyi.irii:i:::j1jri7ZBOS7ivv,
                 ,::,    ::rv77iiiriii:iii:i::,rvLq@huhao.Li
             ,,      ,, ,:ir7ir::,:::i;ir:::i:i::rSGGYri712:
           :::  ,v7r:: ::rrv77:, ,, ,:i7rrii:::::, ir7ri7Lri
          ,     2OBBOi,iiir;r::        ,irriiii::,, ,iv7Luur:
        ,,     i78MBBi,:,:::,:,  :7FSL: ,iriii:::i::,,:rLqXv::
        :      iuMMP: :,:::,:ii;2GY7OBB0viiii:i:iii:i:::iJqL;::
       ,     ::::i   ,,,,, ::LuBBu BBBBBErii:i:i:i:i:i:i:r77ii
      ,       :       , ,,:::rruBZ1MBBqi, :,,,:::,::::::iiriri:
     ,               ,,,,::::i:  @arqiao.       ,:,, ,:::ii;i7:
    :,       rjujLYLi   ,,:::::,:::::::::,,   ,:i,:,,,,,::i:iii
    ::      BBBBBBBBB0,    ,,::: , ,:::::: ,      ,,,, ,,:::::::
    i,  ,  ,8BMMBBBBBBi     ,,:,,     ,,, , ,   , , , :,::ii::i::
    :      iZMOMOMBBM2::::::::::,,,,     ,,,,,,:,,,::::i:irr:i:::,
    i   ,,:;u0MBMOG1L:::i::::::  ,,,::,   ,,, ::::::i:i:iirii:i:i:
    :    ,iuUuuXUkFu7i:iii:i:::, :,:,: ::::::::i:i:::::iirr7iiri::
    :     :rk@Yizero.i:::::, ,:ii:::::::i:::::i::,::::iirrriiiri::,
     :      5BMBBBBBBSr:,::rv2kuii:::iii::,:i:,, , ,,:,:i@petermu.,
          , :r50EZ8MBBBBGOBBBZP7::::i::,:::::,: :,:,::i;rrririiii::
              :jujYY7LS0ujJL7r::,::i::,::::::::::::::iirirrrrrrr:ii:
           ,:  :@kevensun.:,:,,,::::i:i:::::,,::::::iir;ii;7v77;ii;i,
           ,,,     ,,:,::::::i:iiiii:i::::,, ::::iiiir@xingjief.r;7:i,
        , , ,,,:,,::::::::iiiiiiiiii:,:,:::::::::iiir;ri7vL77rrirri::
         :,, , ::::::::i:::i:::i:i::,,,,,:,::i:i:::iir;@Secbone.ii:::

author: Wenyi Chen 
postgraduate of Computer Vision,
Wuhan University, Hubei , China 

email: wenyichen@whu.edu.com or wenyichen550@gmail.com
Date: 2021-05-04 14:52:01
Description: file content
'''
from numpy.core.defchararray import array
from numpy.core.fromnumeric import transpose
from sklearn.metrics.pairwise import paired_distances
import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
import time
import threading
from tqdm import tqdm
import os
import os.path as osp
from PIL import Image
from datasets import Cityscapes
import utils
from segUtil import normalization
from cv2 import cv2
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from utils import my_KMeans_plusplus
from collections import Counter
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
from skimage import io
from utils import KMeans,make_graph_segm_connect_grid2d_conn4,get_neighboring_segments
import glob
import warnings
warnings.filterwarnings("ignore")
softmax = nn.Softmax(dim=1)
class Class_Features:
    def __init__(self, numbers = 19,device=torch.device('cpu')):
        self.class_numbers = numbers
        self.class_features = [[] for i in range(self.class_numbers)]
        self.num = np.zeros(numbers)
        self.num = np.zeros(numbers)
        self.device = device
    def norm(self,vec):
        if len(vec.size()) == 1:
            return torch.sqrt(torch.sum(torch.square(vec)))
        else:
            return torch.sqrt(torch.sum(torch.square(vec),dim=1))

    def process_label(self, label):
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, self.class_numbers + 1, w, h).to(self.device)
        id = torch.where(label < self.class_numbers, label, torch.Tensor([self.class_numbers]).to(self.device))
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1
    
    def allocate_labels(self, feat_cls,objective_vectors,ids, pseudo_label,percentage,img=None):
        feat_proto_distance = self.feat_prototype_distance(feat_cls.unsqueeze(dim=0),objective_vectors,ids)
        feat_proto_distance = feat_proto_distance.squeeze(dim=0).detach().cpu().numpy()
        pseudo_label = pseudo_label.detach().cpu().numpy().astype(np.uint8)
        pseudo_label_copy = pseudo_label.copy()
        #TODO;draw the similarity map
        if img is not None:
            for i,sim_map in enumerate(normalization(feat_proto_distance[ids])):
                # sim_map = normalization(feat_proto_distance)
                heatmap = utils.show_propability_on_image(img,sim_map)
                cv2.imwrite('class_{}_similarity_map.png'.format(ids[i]),heatmap)
        #argmax if use similarity,otherwise argmin
        class_map_by_similarity = np.argmax(feat_proto_distance,axis=0)
        confidence_map_by_similarity = np.amax(feat_proto_distance,axis=0)
        threshold = -np.percentile(-confidence_map_by_similarity, percentage) if percentage < 99 else np.min(confidence_map_by_similarity)-1
        label_mask = pseudo_label == 255
        threshold_mask = confidence_map_by_similarity > threshold
        mask = label_mask * threshold_mask
        pseudo_label_copy[mask] = class_map_by_similarity[mask]
        return pseudo_label_copy

    def feat_prototype_distance(self, feat,objective_vectors,ids=None,similarity='eu'):
        ids_map = {}
        for i,id in enumerate(ids):
            ids_map[id] = i
        N, C, H, W = feat.shape
        feat_proto_similarity = -torch.ones((N, self.class_numbers, H, W)).to(feat.device)
        for i in range(self.class_numbers):
            if ids is not None and i not in ids:
                feat_proto_similarity[:, i, :, :] *= 1e5
            else:
                if(similarity == 'cos'): 
                    a = objective_vectors[ids_map[i]]
                    b = self.norm(a.reshape(-1))
                    a_n = a/ b
                    an = a_n.reshape(-1,1,1).expand_as(feat)
                    feat_n = feat / self.norm(feat)
                    feat_proto_similarity[:, i, :, :] = torch.sum(an * feat_n,dim=1)
                else:
                    # feat -> [256 h w]
                    #objective_vectors[ids_map[i]] -> [256,1,1]
                    feat_proto_similarity[:, i, :, :] = - torch.norm(objective_vectors[ids_map[i]].reshape(-1,1,1).expand(-1, H, W) - feat, 2, dim=1,)

        return feat_proto_similarity

    def calculate_mean_vector_by_output(self, feat_cls, outputs, model):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = model.process_label(outputs_argmax.float())
        outputs_pred = outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def calculate_mean_vector(self, feat_cls, outputs, labels_val=None, model=None):
        a = torch.max(labels_val).item()
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = self.process_label(outputs_argmax.float())
        if labels_val is None:
            outputs_pred = outputs_argmax
        else:
            labels_expanded = self.process_label(labels_val)
            outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(t)
        return vectors, ids

class STAGE2():
    def __init__(self,round_idx, args,save_round_eval_path=None, save_sp_path = None):
        self.round_idx = round_idx
        self.class_numbers = args.num_classes
        self.objective_vectors = torch.zeros([self.class_numbers, 2048])
        self.objective_vectors_num = torch.zeros([self.class_numbers]) 
        # self.proto_momentum = args.proto_momentum
        self.source_dataset = args.source_dataset
        self.target_dataset = args.target_dataset
        if save_round_eval_path is not None:
            self.save_round_eval_path = save_round_eval_path
        else:
            self.save_round_eval_path = osp.join(args.save_result_path,'round_{}_our_light'.format(self.round_idx))
        self.save_prototype_path = osp.join(self.save_round_eval_path, 'prototype')
        self.save_processed_labels_path = osp.join(self.save_round_eval_path, 'processed_cluster_labels')
        self.save_processed_labels_color_path = osp.join(self.save_round_eval_path, 'processed_cluster_labels_color')
        self.save_multiview_labels_color_path = osp.join(self.save_round_eval_path, '{}_muti_views_labels_color'.format(args.seg_num))
        # self.save_multiview_labels_path = osp.join(self.save_round_eval_path, 'muti_views_labels')
        # self.save_multiview_labels_intra_path = osp.join(self.save_round_eval_path, 'muti_views_labels_intra')
        self.save_multiview_labels_path = osp.join(self.save_round_eval_path, '{}_muti_views_labels'.format(args.seg_num))
        self.save_multiview_labels_intra_path = osp.join(self.save_round_eval_path, '{}_muti_views_labels_intra'.format(args.seg_num))
        if save_sp_path is not None:
            self.save_sp_path = save_sp_path
        else:
            self.save_sp_path = osp.join(args.save_result_path, 'superpixels')
        self.denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images
        if not os.path.exists(self.save_prototype_path):
            os.mkdir(self.save_prototype_path)
        if not os.path.exists(self.save_processed_labels_path):
            os.mkdir(self.save_processed_labels_path)
        if not os.path.exists(self.save_processed_labels_color_path):
            os.mkdir(self.save_processed_labels_color_path)
        if not os.path.exists(self.save_multiview_labels_color_path):
            os.mkdir(self.save_multiview_labels_color_path)
        if not os.path.exists(self.save_multiview_labels_path):
            os.mkdir(self.save_multiview_labels_path)
        if not os.path.exists(self.save_multiview_labels_intra_path):
            os.mkdir(self.save_multiview_labels_intra_path)
        if not os.path.exists(self.save_sp_path):
            os.mkdir(self.save_sp_path)

        self.objective_vectors_fog0005 = torch.zeros([self.class_numbers, 2048])
        self.objective_vectors_num_fog0005 = torch.zeros([self.class_numbers]) 
        self.objective_vectors_fog001 = torch.zeros([self.class_numbers, 2048])
        self.objective_vectors_num_fog001 = torch.zeros([self.class_numbers]) 
        self.objective_vectors_fog002 = torch.zeros([self.class_numbers, 2048])
        self.objective_vectors_num_fog002 = torch.zeros([self.class_numbers]) 


    def update_objective_SingleVector(self, id, vector, name='moving_average', start_mean=True):
        if vector.sum().item() == 0:
            return
        if start_mean and self.objective_vectors_num[id].item() < 100:
            name = 'mean'
        if name == 'moving_average':
            self.objective_vectors[id] = self.objective_vectors[id] * (1 - self.proto_momentum) + self.proto_momentum * vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
        elif name == 'mean':
            self.objective_vectors[id] = self.objective_vectors[id] * self.objective_vectors_num[id] + vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors[id] = self.objective_vectors[id] / self.objective_vectors_num[id]
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
            pass
        else:
            raise NotImplementedError('no such updating way of objective vectors {}'.format(name))

    def update_objective_SingleVector2(self, id, vector,objective_vectors,objective_vectors_num, name='moving_average', start_mean=True):
        if vector.sum().item() == 0:
            return
        if start_mean and objective_vectors_num[id].item() < 100:
            name = 'mean'
        if name == 'moving_average':
            objective_vectors[id] = objective_vectors[id] * (1 - self.proto_momentum) + self.proto_momentum * vector.squeeze()
            objective_vectors_num[id] += 1
            objective_vectors_num[id] = min(objective_vectors_num[id], 3000)
        elif name == 'mean':
            objective_vectors[id] = objective_vectors[id] * objective_vectors_num[id] + vector.squeeze()
            objective_vectors_num[id] += 1
            objective_vectors[id] = objective_vectors[id] / objective_vectors_num[id]
            objective_vectors_num[id] = min(objective_vectors_num[id], 3000)
            pass
        else:
            raise NotImplementedError('no such updating way of objective vectors {}'.format(name))

    def calc_prototype_and_cluster(self,opt,loader,model,device,logger):
        '''
        @description: calculate prototypes for several large classes in one target image, and allocate
         labels for low confidence pixels via the similarity from the pixels to each prototype
        @param {*} opt
        @param {*} loader
        @param {*} model
        @param {*} device
        @param {*} logger
        @return {*}
        '''    
        class_features = Class_Features(numbers=self.class_numbers,device=device)

        for i, datas in tqdm(enumerate(loader)):
            if(len(datas)==3):
                images,labels,sample_names = datas
            elif(len(datas) == 2):
                images,_ = datas
            else:
                print("output shape from dataloader is not correct")
                return
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long) #pseudo label
            model = model.to(device)
            
            model.eval()
            with torch.no_grad():
                out,feat = model(images)
                batch, w, h = labels.size()
                newlabels = labels.reshape([batch, 1, w, h]).float()
                newlabels = F.interpolate(newlabels, size=feat.size()[2:], mode='nearest')
                images = F.interpolate(images, size=feat.size()[2:], mode='nearest')
                images = images.detach().cpu().numpy()
                for i in range(len(images)):
                    ofeat = feat[i]
                    oout = out[i]
                    sample_name = sample_names[i].split('/')[-1]
                    sample_name = sample_name[:sample_name.rfind('.')]
                    newlabel = newlabels[i]
                    newlabel_cp = newlabel.squeeze(dim=0).detach().cpu().numpy().astype(np.int64)
                    origin_label_d = Cityscapes.decode_target(newlabel_cp).astype(np.uint8)
                    Image.fromarray(origin_label_d).save('%s/%s_color_origin.png' % (self.save_processed_labels_color_path, sample_name))
                    img = self.denorm(images[i]).transpose(1,2,0)
                    for percentage in [50,100]:
                        vectors, ids = class_features.calculate_mean_vector(ofeat.unsqueeze(dim=0), oout.unsqueeze(dim=0),
                            newlabel.unsqueeze(dim=0), model)
                        newlabel = newlabel.squeeze(dim=0)
                        processed_label = class_features.allocate_labels(ofeat,vectors,ids,newlabel,percentage,img=img)
                        processed_label_d = Cityscapes.decode_target(processed_label.copy()).astype(np.uint8)
                        Image.fromarray(processed_label).save('%s/%s_%d.png' % (self.save_processed_labels_path, sample_name,percentage))
                        Image.fromarray(processed_label_d).save('%s/%s_%d_color.png' % (self.save_processed_labels_color_path, sample_name,percentage))
                        newlabel = torch.from_numpy(processed_label).float().to(device)
                        newlabel = newlabel.unsqueeze(dim=0)


    def calc_global_prototype(self,opt,loader,issource,model,device,logger):
        class_features = Class_Features(numbers=self.class_numbers,device=device)

        # begin training
        for epoch in range(opt.proto_epoch):
            for i, datas in tqdm(enumerate(loader)):
                if(len(datas)==3):
                    images,labels,sample_names = datas
                elif(len(datas) == 2):
                    images,_ = datas
                else:
                    print("output shape from dataloader is not correct")
                    return

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                model = model.to(device)
                
                model.eval()
                if issource: #source->use a part of predict result as pesudo label
                    with torch.no_grad():
                        out,feat = model(images)
                        batch, w, h = labels.size()
                        newlabels = labels.reshape([batch, 1, w, h]).float()
                        newlabels = F.interpolate(newlabels, size=feat.size()[2:], mode='nearest')
                        vectors, ids = class_features.calculate_mean_vector(feat, out, newlabels, model)
                        for t in range(len(ids)):
                            self.update_objective_SingleVector(ids[t], vectors[t].detach().cpu().numpy(), 'mean')
                else: #target->use the whole predict result as pesudo label
                    with torch.no_grad():
                        out,feat = model(images)
                        vectors, ids = class_features.calculate_mean_vector(feat, out, model=model)
                        #vectors, ids = class_features.calculate_mean_vector_by_output(feat_cls, output, model)
                        for t in range(len(ids)):
                            self.update_objective_SingleVector(ids[t], vectors[t].detach().cpu(), 'mean')

        if issource:
            save_path = os.path.join(os.path.dirname(self.save_prototype_path), "prototypes_on_{}".format(self.source_dataset))
        else:
            save_path = os.path.join(os.path.dirname(self.save_prototype_path), "prototypes_on_{}".format(self.target_dataset))
        torch.save(self.objective_vectors, save_path)

    def cluster_use_global_prototype(self,opt,loader,issource,model,device,logger):
        '''
        @description:  allocate labels for low confidence pixels via the similarity from the pixels 
        to each global class prototype
        @param {*} opt
        @param {*} loader
        @param {*} model
        @param {*} device
        @param {*} logger
        @return {*}
        '''    
        if issource:
            save_path = os.path.join(os.path.dirname(self.save_prototype_path), "prototypes_on_{}".format(self.source_dataset))
        else:
            save_path = os.path.join(os.path.dirname(self.save_prototype_path), "prototypes_on_{}".format(self.target_dataset))
        self.objective_vectors = torch.load(save_path).to(device)

        for i, datas in tqdm(enumerate(loader)):
            if(len(datas)==3):
                images,labels,sample_names = datas
            elif(len(datas) == 2):
                images,_ = datas
            else:
                print("output shape from dataloader is not correct")
                return

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long) #pseudo label
            model = model.to(device)
            ids = np.array(range(len(self.objective_vectors)))
            class_features = Class_Features(numbers=self.class_numbers,device=device)
            model.eval()
            with torch.no_grad():
                out,feat = model(images)
                batch, w, h = labels.size()
                newlabels = labels.reshape([batch, 1, w, h]).float()
                newlabels = F.interpolate(newlabels, size=feat.size()[2:], mode='nearest')
                images = F.interpolate(images, size=feat.size()[2:], mode='nearest')
                images = images.detach().cpu().numpy()
                for i in range(len(images)):
                    ofeat = feat[i]
                    oout = out[i]
                    sample_name = sample_names[i].split('/')[-1]
                    sample_name = sample_name[:sample_name.rfind('.')]
                    newlabel = newlabels[i]
                    newlabel_cp = newlabel.squeeze(dim=0).detach().cpu().numpy().astype(np.int64)
                    origin_label_d = Cityscapes.decode_target(newlabel_cp).astype(np.uint8)
                    Image.fromarray(origin_label_d).save('%s/%s_color_origin.png' % (self.save_processed_labels_color_path, sample_name))
                    img = self.denorm(images[i]).transpose(1,2,0)

                    for percentage in [50,100]:
                        newlabel = newlabel.squeeze(dim=0)
                        processed_label = class_features.allocate_labels(ofeat,self.objective_vectors,ids,newlabel,percentage,img=img)
                        processed_label_d = Cityscapes.decode_target(processed_label.copy()).astype(np.uint8)
                        Image.fromarray(processed_label).save('%s/%s_%d.png' % (self.save_processed_labels_path, sample_name,percentage))
                        Image.fromarray(processed_label_d).save('%s/%s_%d_color.png' % (self.save_processed_labels_color_path, sample_name,percentage))
                        newlabel = torch.from_numpy(processed_label).float().to(device)
                        newlabel = newlabel.unsqueeze(dim=0)

    def extend_pseudo_by_kmeans(self,opt,loader,model,device,proportion=0.5):

        # begin training
        for i, datas in enumerate(tqdm(loader)):
            if(len(datas)==3):
                images,labels,sample_names = datas
            elif(len(datas) == 2):
                images,_ = datas
            else:
                print("output shape from dataloader is not correct")
                return

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            model = model.to(device)
            
            model.eval()
            with torch.no_grad():
                outs,feats = model(images)
                outs = softmax(outs)
                #NOTE resnet特征需要使用
                # feats = F.interpolate(feats, size=outs.size()[2:], mode='bilinear')
                batch, w_o, h_o = labels.size()
                newlabels = labels.reshape([batch, 1, w_o, h_o]).float()
                newlabels = F.interpolate(newlabels, size=feats.size()[2:], mode='nearest').detach().cpu().numpy()
                newlabels = newlabels[0].astype(np.uint8)
                ulabels = newlabels==255
                # a = np.unique(newlabels)
                h,w = newlabels[0].shape
                indexs = [np.array(range(w))+i*w for i in range(h)]
                indexs = np.array(indexs)
                probs = outs.detach().cpu().numpy()
                feats = feats.detach().cpu().numpy()
                for t in range(len(feats)):
                    sample_name = sample_names[t].split('/')[-1]
                    sample_name = sample_name[:sample_name.rfind('.')]
                    prob = probs[t]
                    pred = np.argmax(prob,axis=0)
                    feat = feats[t]
                    ulabel = ulabels[t]
                    newlabel = newlabels[t]
                    newlabel_cp = newlabel.copy()
                    conf = np.max(prob,axis=0)
                    entropy = -np.sum( prob * np.log(prob+np.finfo(float).eps), axis= 0 )
                    weight = (1 - conf) + entropy
                    ulabel_w = weight[ulabel]
                    uindex = indexs[ulabel]
                    ufeat = feat[:,ulabel]
                    cluset_y,_,val_samples = my_KMeans_plusplus(ufeat.transpose(1,0),K=10,weight = ulabel_w,proportion = proportion)
                    val_index = uindex[val_samples]
                    v_mask = np.zeros_like(indexs,dtype=bool)
                    a = np.unique(val_index)
                    v_mask.ravel()[val_index] = True
                    newlabel[v_mask] = pred[v_mask]
                    # t[val_samples] = ulabel_origin[val_samples]
                    pred_label_labelIDs = Cityscapes.train_id_to_id[newlabel.copy()]
                    pred_label_labelIDs = cv2.resize(pred_label_labelIDs,(h_o,w_o),interpolation=cv2.INTER_NEAREST) 
                    Image.fromarray(pred_label_labelIDs.astype(np.uint8)).save('%s/%s.png' % (self.save_processed_labels_path, sample_name))
                    processed_label_d = Cityscapes.decode_target(newlabel.copy()).astype(np.uint8)
                    newlabel_cp_d = Cityscapes.decode_target(newlabel_cp).astype(np.uint8)
                    img_perceptual = cv2.hconcat([newlabel_cp_d, processed_label_d])  # 水平拼接
                    Image.fromarray(img_perceptual).save('%s/%s_%d_color.png' % (self.save_processed_labels_color_path, sample_name,proportion))
                    Image.fromarray(processed_label_d).save('%s/%s_%d_color2.png' % (self.save_processed_labels_color_path, sample_name,proportion))
                    # Image.fromarray((normalization(entropy)*255).astype(np.uint8)).save('%s/%s_%d_entropy.png' % (self.save_processed_labels_color_path, sample_name,proportion))

    def extend_pseudo_by_superpixels_m(self,opts,plabel_path,confidence_map_path,images_path=None):
        print('###### Start extend pseudo label by superpixels in round {} ! ######'.format(self.round_idx))
        start_pl = time.time()
        thread_nums = 6
        threads_pools = []
        plabel_list = glob.glob(os.path.join(plabel_path,'*.png'))
        slice_size = int(len(plabel_list) / thread_nums)
        for i in range(thread_nums):
            _range = (i*slice_size,(i+1)*slice_size) if i != thread_nums - 1 else (i*slice_size,len(plabel_list))
            t = threading.Thread(target=self.extend_pseudo_by_superpixels, args=(opts,plabel_path,confidence_map_path,images_path,i,_range))
            threads_pools.append(t)
            t.start()
        for thread in threads_pools:
            thread.join()
        print('###### end extend pseudo label by superpixels in round {} ! Time cost: {:.2f} seconds. ######'.format(self.round_idx,time.time() - start_pl))


    def extend_pseudo_by_superpixels(self,opts,plabel_path,confidence_map_path,images_path=None,thread_id=0,_range=None):
        plabel_list = glob.glob(os.path.join(plabel_path,'*.png'))
        utime = 0
        if _range is not None:
            plabel_list = plabel_list[_range[0]:_range[1]]
        
        for i,plabel_name in enumerate(plabel_list):
            if thread_id == 0 and  i % 100 == 0:
                print("done {} iteration, total:{}".format(i,len(plabel_list)))
            plabel = Cityscapes.encode_target(np.array(Image.open(plabel_name)))
            sample_name_o = os.path.basename(plabel_name)
            sample_name = sample_name_o.split('.png')[0]
            #划分超像素
            if opts.save_val_results:
                pred_d = Cityscapes.decode_target(plabel.copy()).astype(np.uint8)
                Image.fromarray(pred_d).save('%s/%s_plabel.png' % (self.save_multiview_labels_color_path, sample_name))
            sp_save_path = os.path.join(self.save_sp_path,'{}.npy'.format(sample_name))
            segments = None
            if os.path.exists(sp_save_path):
                segments = np.load(sp_save_path)
            else:
                #print('re segement')
                if images_path is None:
                    raise('target image does\'nt exist in {}'.format(images_path))
                image_path = os.path.join(images_path,sample_name_o)
                image = Image.open(image_path)
                start_time = time.time()
                segments = slic(image, n_segments = opts.seg_num, sigma = 1)
                end_time = time.time()
                utime += round(1000 * (end_time - start_time))
                np.save(sp_save_path,segments)
            # io.imsave('%s/%s_segs.png' % (self.save_multiview_labels_color_path, sample_name), (mark_boundaries(image , segments)* 255).astype(np.uint8))
            #读取confidence_map
            if not os.path.exists(os.path.join(confidence_map_path,sample_name_o)):
                continue
            confidence_map = np.array(Image.open(os.path.join(confidence_map_path,sample_name_o)))
            confidence_map = Cityscapes.encode_target(confidence_map)

            #超像素内部划分
            uniq_segs = np.unique(segments)
            ava = plabel != 255
            noava = plabel == 255
            class_indexs = []
            for c in range(opts.num_classes):
                class_indexs.append(confidence_map == c)
            for seg_id in uniq_segs:
                seg_index = segments == seg_id
                supepixel_mask = seg_index & ava
                domain_class = np.unique(plabel[supepixel_mask])
                if len(domain_class) == 0:
                    continue
                for c in domain_class:
                    ind = class_indexs[c] & seg_index & noava
                    plabel[ind] = c
            if opts.save_val_results:
                pred_d = Cityscapes.decode_target(plabel.copy()).astype(np.uint8)
                Image.fromarray(pred_d).save('%s/%s_process_intra.png' % (self.save_multiview_labels_color_path, sample_name))
            pred_label_labelIDs = Cityscapes.train_id_to_id[plabel.copy()]
            Image.fromarray(pred_label_labelIDs.astype(np.uint8)).save('%s/%s.png' % (self.save_multiview_labels_intra_path, sample_name))
            # change_mask = (plabel_o - plabel) != 0
            # Image.fromarray(change_mask.astype(np.uint8)).save('%s/%s_change.png' % (self.save_multiview_labels_intra_path, sample_name))
        # utime /= len(plabel_list)
        print('###### mean time for generate sp: {} ! '.format(utime))

def generateSuperpiexls(imgs_path,save_path):
    print('###### Start superpixels generation ######')
    start_pl = time.time()
    ## output of deeplab is logits, not probability
    imgs_list = glob.glob(os.path.join(imgs_path,'*.png'))
    with torch.no_grad():
        for i,image_path in enumerate(tqdm(imgs_list)):
            sample_name_o = os.path.basename(image_path)
            sample_name = sample_name_o.split('.png')[0]
            image = Image.open(image_path)
            #划分超像素
            segments = None
            segments = slic(image, n_segments = 500, sigma = 1)
            sp_save_path = os.path.join(save_path,'{}.npy'.format(sample_name))
            np.save(sp_save_path,segments)
    print('###### Finish superpixels generation! Time cost: {:.2f} seconds. ######'.format(time.time() - start_pl))
                

if __name__ == '__main__':
    imgs_path = 'datasets/targetImgs'
    save_path = 'results/superpixels'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    generateSuperpiexls(imgs_path,save_path=save_path)