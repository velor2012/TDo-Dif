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
Date: 2021-04-29 20:27:45
Description: generate the psudel labels with high confidence base on CRST https://github.com/yzou2/CRST
'''
import numpy as np
import os
from cv2 import cv2
from numpy.core.numeric import zeros_like
import torch
from packaging import version
from torch.nn.functional import threshold
from tqdm import tqdm
import time
import utils
import math
from datasets import Cityscapes
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt 
import logging
import threading
lock = threading.Lock()  # 同步锁
import torch.nn.functional as F
import glob
import multiprocessing as mp
class STAGE1():
    def __init__(self,round_idx, args,save_round_eval_path=None):
        self.round_idx = round_idx
        self.tgt_portion = args.init_target_portion 
        # self.max_tgt_port = args.max_target_portion
        self.tgt_port_step = args.target_portion_step
        if save_round_eval_path is not None:
            self.save_round_eval_path = save_round_eval_path
        else:
            self.save_round_eval_path = osp.join(args.save_result_path,'round_{}_our_light'.format(self.round_idx))
        self.save_origin_wpred_path = osp.join(self.save_round_eval_path, 'origin_wpred')
        self.save_origin_pred_path = osp.join(self.save_round_eval_path, 'origin_pred')
        if args.target_dataset == 'FoggyZurich':
            self.save_img_path = osp.join('datasets/targetImgs')
        elif args.target_dataset == 'ACDC':
            self.save_img_path = osp.join('datasets/acdcImgs')
        else:
            self.save_img_path = osp.join('datasets/drivingImgs')
        self.save_confidence_prob_path = osp.join(self.save_round_eval_path, 'conf_prob')
        self.save_confidence_pred_path = osp.join(self.save_round_eval_path, 'conf_pred')
        self.save_confidence_pred_color_path = osp.join(self.save_round_eval_path, 'conf_color')
        self.save_wpred_vis_path = osp.join(self.save_round_eval_path,'weighted_pred_vis')
        self.save_soft_label_path = osp.join(self.save_round_eval_path,'soft_labels_init')
        self.save_pseudo_label_weighted_color_path =  osp.join(self.save_round_eval_path,'pseudo_label_weighted_color')
        self.save_pseudo_label_weighted_path = osp.join(self.save_round_eval_path, 'pseudo_label_weighted')
        
        self.save_stats_path = osp.join(self.save_round_eval_path, 'stats')
        self.denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images
        self.interp = None

        # #add position prior
        # if args.useprior:
        #     self.prior_map = {}
        #     for file_name in os.listdir(args.prior_map_dir):
        #         t = file_name.split('position_class_')[-1]
        #         class_id = int(t[:t.find('.')])
        #         self.prior_map[class_id] = cv2.imread(osp.join(args.prior_map_dir,file_name),0)

        #         #draw the prior map value

        if not os.path.exists(args.save_result_path):
            os.mkdir(args.save_result_path)
        if not os.path.exists(self.save_round_eval_path):
            os.mkdir(self.save_round_eval_path)
        if not os.path.exists(self.save_origin_pred_path):
            os.mkdir(self.save_origin_pred_path)
        if not os.path.exists(self.save_origin_wpred_path):
            os.mkdir(self.save_origin_wpred_path)
        if not os.path.exists(self.save_confidence_prob_path):
            os.mkdir(self.save_confidence_prob_path)
        if not os.path.exists(self.save_confidence_pred_path):
            os.mkdir(self.save_confidence_pred_path)
        if not os.path.exists(self.save_img_path):
            os.mkdir(self.save_img_path)
        if not os.path.exists(self.save_confidence_pred_color_path):
            os.mkdir(self.save_confidence_pred_color_path)
        if not os.path.exists(self.save_wpred_vis_path):
            os.mkdir(self.save_wpred_vis_path)
        if not os.path.exists(self.save_soft_label_path):
            os.mkdir(self.save_soft_label_path)
        if not os.path.exists(self.save_pseudo_label_weighted_color_path):
            os.mkdir(self.save_pseudo_label_weighted_color_path)
        if not os.path.exists(self.save_pseudo_label_weighted_path):
            os.mkdir(self.save_pseudo_label_weighted_path)
        if not os.path.exists(self.save_stats_path):
            os.mkdir(self.save_stats_path)
        ## upsampling layer
        self.interp = None
    def cal_class_wise_confidence_infor(self,model, device, loader, args):
        return self.cal_class_wise_confidence_infor_ov(model, device, loader, args)

    def cal_class_wise_confidence_infor_ov(self,model, device, loader, args):
        '''
        @description: 
        @param {*} self
        @param {*} model
        @param {*} device
        @param {*} loader
        @param {*} args
        @return {*}  conf_dict, pred_cls_num
        '''    
        print('###### Start collecting conf infor in round {} ! ######'.format(self.round_idx))
        # saving output data
        conf_dict = {k: [] for k in range(args.num_classes)}
        pred_cls_num = np.zeros(args.num_classes)

        ## output of deeplab is logits, not probability
        softmax2d = torch.nn.Softmax2d()
        img_id = 0
        with torch.no_grad():
            for i, datas in enumerate(loader):
                if i % 100 == 0:
                    print("done {} iteration, total:{}".format(i,len(loader.dataset)))
                images = datas['images']
                # labels = datas['labels']
                if 'img_paths' in datas:
                    sample_names = datas['img_paths']

                images = images.to(device, dtype=torch.float32)
                outputs = model(images)[0]

                if self.interp is None:
                    self.interp = torch.nn.Upsample(size=images.shape[2:], mode='bilinear')
                # outputs_interpolated = softmax2d(self.interp(outputs)).cpu().numpy()
                outputs1 = softmax2d(outputs).cpu().numpy()
                outputs = outputs.cpu().numpy()
                # preds_up = np.argmax(outputs_interpolated,axis=1)
                # confs = np.amax(outputs_interpolated,axis=1)
                preds = np.argmax(outputs1,axis=1)
                confs = np.amax(outputs1,axis=1)
                for j in range(len(images)):
                    pred = preds[j]
                    # pred_up = preds_up[j]
                    conf = confs[j]
                    prob = outputs[j]

                    # 保存soft label
                    t = os.path.basename(sample_names[j])
                    t = t[:t.rfind('.')]
                    np.save(osp.join(self.save_soft_label_path, t), prob)

                    # if args.save_val_results and self.save_img_path:
                    if not osp.exists('{}/{}'.format(self.save_img_path, os.path.basename(sample_names[j]))):
                        image = images[j].detach().cpu().numpy()
                        image = (self.denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                        Image.fromarray(image).save('%s/%s' % (self.save_img_path, os.path.basename(sample_names[j])))
                    # # save class-wise confidence maps
                    for idx_cls in range(args.num_classes):
                        idx_temp = pred == idx_cls
                        pred_cls_num[idx_cls] += np.sum(idx_temp)
                        if idx_temp.any():
                            conf_cls_temp = conf[idx_temp].astype(np.float32)
                            len_cls_temp = conf_cls_temp.size
                            # downsampling by ds_rate
                            conf_cls = conf_cls_temp[0:len_cls_temp:args.ds_rate]
                            conf_dict[idx_cls].extend(conf_cls)
                    img_id += 1
        print('###### end collecting conf infor in round {} ! ######'.format(self.round_idx))
        return conf_dict, pred_cls_num

    def cal_threshold_kc(self,conf_dict, pred_cls_num, args):
        '''
        @description: 
        @param {*} self
        @param {*} conf_dict
        @param {*} pred_cls_num
        @param {*} args
        @return {np.array with len num_class} cls_thresh
        '''    
        print('###### Start kc generation in round {} ! ######'.format(self.round_idx))
        # self.update_pseudo_portion()
        print('###### tgt_portion in round {} : {} ! ######'.format(self.round_idx,self.tgt_portion))
        start_kc = time.time()
        # threshold for each class
        cls_thresh = np.ones(args.num_classes,dtype = np.float32)
        cls_sel_size = np.zeros(args.num_classes, dtype=np.float32)
        cls_size = np.zeros(args.num_classes, dtype=np.float32)
        for idx_cls in np.arange(0, args.num_classes):
            cls_size[idx_cls] = pred_cls_num[idx_cls]
            if conf_dict[idx_cls] != None:
                conf_dict[idx_cls].sort(reverse=True) # sort in descending order
                len_cls = len(conf_dict[idx_cls])
                cls_sel_size[idx_cls] = int(math.floor(len_cls * self.tgt_portion))
                len_cls_thresh = int(cls_sel_size[idx_cls])
                if len_cls_thresh != 0:
                    cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh-1]
                conf_dict[idx_cls] = None
        
        np.save(self.save_stats_path + '/cls_thresh_round' + str(self.round_idx) + '.npy', cls_thresh)
        np.save(self.save_stats_path + '/cls_sel_size_round' + str(self.round_idx) + '.npy', cls_sel_size)
        print('###### Finish kc generation in round {}! Time cost: {:.2f} seconds. ######'.format(self.round_idx,time.time() - start_kc))
        return cls_thresh

    # def update_pseudo_portion(self):
    #     self.tgt_portion = min(self.tgt_portion + self.tgt_port_step*self.round_idx, self.max_tgt_port)

    def label_selection(self,model,loaders,cls_thresh,device, args):
        print('###### Start pseudo-label generation in round {} ! ######'.format(self.round_idx))
        start_pl = time.time()
        process_num = 4 #四进程
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(process_num) # 7.7G
        for thread_id,loader in enumerate(loaders):
            if thread_id == 0:
                continue
            pool.apply_async(self.label_selection_one_view, args=(model,loader,cls_thresh,device,thread_id, args))
        self.label_selection_one_view(model,loaders[0],cls_thresh,device,0, args)
        pool.close()
        pool.join()
        print('###### Finish pseudo-label generation in round {}! Time cost: {:.2f} seconds. ######'.format(self.round_idx,time.time() - start_pl))

    # def muti_thread_lb(self,model,loaders,cls_thresh,device, args):
    #     threads_pools = []
    #     for thread_id,loader in enumerate(loaders):
    #         t = threading.Thread(target=self.label_selection_one_view, args=(model,loader,cls_thresh,device,thread_id, args))
    #         t.start()
    #         threads_pools.append(t)
    #     for thread in threads_pools:
    #         thread.join()
    #     return 0

    def label_selection_one_view(self,model,loader,cls_thresh,device,thread_id, args):

        ## output of deeplab is logits, not probability
        softmax2d = torch.nn.Softmax2d()
        with torch.no_grad():
            for i, datas in enumerate(loader):
            # for i, datas in enumerate(tqdm(loader)):
                images = datas['images']
                # labels = datas['labels']
                if 'img_paths' in datas:
                    sample_names = datas['img_paths']
                images = images.to(device, dtype=torch.float32)
                # lock.acquire()
                if thread_id == 0 and i % 100 == 0:
                    print("done {} iteration, total:{}".format(i,len(loader.dataset)))
                outputs = model(images)[0]
                # lock.release()
                # if self.interp is None:
                self.interp = torch.nn.Upsample(size=images.shape[2:], mode='bilinear')
                
                outputs_interpolated = softmax2d(self.interp(outputs)).cpu().numpy()

                for i in range(len(outputs_interpolated)):
                    pred_prob = outputs_interpolated[i]
                    sample_name = sample_names[i].split('/')[-1]
                    sample_name = sample_name[:sample_name.rfind('.')]

                    weighted_prob = pred_prob.transpose(1,2,0)/cls_thresh
                    weighted_prob = weighted_prob.transpose(2,0,1)
                    weighted_pred_trainIDs = np.asarray(np.argmax(weighted_prob, axis=0), dtype=np.uint8)
                    wpred_label_col = weighted_pred_trainIDs.copy()
                    pred_label_labelIDs = Cityscapes.train_id_to_id[wpred_label_col].astype(np.uint8)
                    Image.fromarray(pred_label_labelIDs).save('%s/%s.png' % (self.save_origin_wpred_path, sample_name))
                    if(args.save_val_results):
                        #save weighted predication
                        wpred_label_col = Cityscapes.decode_target(weighted_pred_trainIDs.copy()).astype(np.uint8)
                        Image.fromarray(wpred_label_col).save('%s/%s_color.png' % (self.save_wpred_vis_path, sample_name))
                    weighted_conf = np.amax(weighted_prob, axis=0)
                    pred_label_trainIDs = weighted_pred_trainIDs.copy()
                    pred_label_labelIDs = Cityscapes.train_id_to_id[pred_label_trainIDs]
                    pred_label_labelIDs[weighted_conf < 1] = 0  # '0' in cityscapes indicates 'unlabaled' for labelIDs
                    pred_label_trainIDs[weighted_conf < 1] = 255 # '255' in cityscapes indicates 'unlabaled' for trainIDs


                    # pseudo-labels with labelID
                    pseudo_label_labelIDs = pred_label_labelIDs.copy()
                    pseudo_label_trainIDs = pred_label_trainIDs.copy()
                    if(args.save_val_results):
                        # save colored pseudo-label map
                        pseudo_label_col = Cityscapes.decode_target(pseudo_label_trainIDs).astype(np.uint8)
                        Image.fromarray(pseudo_label_col).save('%s/%s_color.png' % (self.save_pseudo_label_weighted_color_path, sample_name))
                
                    # save pseudo-label map with label IDs
                    pseudo_label_save = Image.fromarray(pseudo_label_labelIDs.astype(np.uint8))
                    pseudo_label_save.save('%s/%s.png' % (self.save_pseudo_label_weighted_path, sample_name))

