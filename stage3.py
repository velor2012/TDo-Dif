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
from operator import le
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
import torch.nn.functional as F
import glob
from utils import *
from natsort import natsorted
import warnings

import multiprocessing as mp

warnings.filterwarnings("ignore")
class STAGE3():
    def __init__(self,round_idx, args,save_round_eval_path=None):
        from GLUnet.refine_pseudo import refineSeg
        self.refineSeg = refineSeg
        self.round_idx = round_idx
        self.tgt_portion = args.init_target_portion 
        self.target_portion_step = args.target_portion_step
        if save_round_eval_path is not None:
            self.save_round_eval_path = save_round_eval_path
        else:
            self.save_round_eval_path = osp.join(args.save_result_path,'round_{}_our_light'.format(self.round_idx))
        self.save_proccess_label_path = osp.join(self.save_round_eval_path, 'stage3_proccess_labelfew')
        self.save_proccess_label_color_path = osp.join(self.save_round_eval_path, 'stage3_proccess_label_colorfew')
        self.save_concat_label_path = osp.join(self.save_round_eval_path, 'concat_label_few')
        self.save_concat_label_color_path = osp.join(self.save_round_eval_path, 'concat_label_color_few')
        
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

        if not os.path.exists(self.save_round_eval_path):
            os.mkdir(self.save_round_eval_path)
        if not os.path.exists(self.save_proccess_label_path):
            os.mkdir(self.save_proccess_label_path)
        if not os.path.exists(self.save_proccess_label_color_path):
            os.mkdir(self.save_proccess_label_color_path)
        if not os.path.exists(self.save_concat_label_path):
            os.mkdir(self.save_concat_label_path)
        if not os.path.exists(self.save_concat_label_color_path):
            os.mkdir(self.save_concat_label_color_path)
    
    def del_GLU(self):
        del self.GLUNet

    def label_propagation(self,model,device,loaders,flow_dir,args):
        print('###### Start label propagation in round {} ! ######'.format(self.round_idx))
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(2) # 7.7G
        for thread_id,loader in enumerate(loaders):
            if thread_id == 0:
                continue
            pool.apply_async(self.label_propagation_new, args=(model,device,loader,flow_dir,args,thread_id))
        self.label_propagation_new(model,device,loaders[0],flow_dir,args,0)
        pool.close()
        pool.join()

    #loader必须有超像素扩充之后的标签
    def label_propagation_new(self,model,device,loader,flow_dir,args,thread_id = 0):
        start_pl = time.time()
        ## output of deeplab is logits, not probability
        softmax2d = torch.nn.Softmax2d()
        with torch.no_grad():
            #遍历顺序为从近景到远景
            closer_img_path,closer_img_name,closer_img,closer_pred,closer_prob = None,None,None,None,None
            prob,pred,closer_plabel,proccessed_pred = None,None,None,None
            cc = 0
            for i, datas in enumerate(tqdm(loader)):
                if thread_id == 0 and  i % 100 == 0:
                    print("done {} iteration, total:{}".format(i,len(loader.dataset)))
                images = datas['images']
                labels = datas['labels']
                if 'img_paths' in datas:
                    sample_names = datas['img_paths']
                images = images.to(device, dtype=torch.float32)
                # labels = labels.to(device, dtype=torch.uint8)
                # lock2.acquire()
                outputs,_ = model(images)
                # lock2.release()
                if self.interp is None:
                    self.interp = torch.nn.Upsample(size=images.shape[2:], mode='bilinear')
                outputs_interpolated = softmax2d(self.interp(outputs))
                for j in range(len(outputs_interpolated)):
                    prob = outputs_interpolated[j]
                    prob = prob.permute(1,2,0)
                    sample_base_name = os.path.basename(sample_names[j])
                    sample_name = sample_base_name[:sample_base_name.rfind('.')]
                    label_f = labels[j]
                    a = torch.sum(label_f == 255)
                    label_f_n = label_f.numpy()
                    temp_string = sample_name
                    # image = images[j]
                    # image = image.detach().cpu().numpy()
                    # image_d  = (self.denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    if(closer_img_path is None):
                        cc += 1
                        Image.fromarray(Cityscapes.train_id_to_id[label_f_n].astype(np.uint8)).save('%s/%s.png' % (self.save_concat_label_path,temp_string))
                    else:
                        far_img_id = temp_string[-6:]
                        temp_string2 = os.path.basename(closer_img_path)
                        temp_string2 = temp_string2.split('.png')[0]
                        closer_img_id = temp_string2[-6:]
                        a = int(far_img_id)
                        b = int(closer_img_id)
                        c = os.path.basename(temp_string)[:10]
                        d = os.path.basename(closer_img_path)[:10] 
                        if(a == b - 1 and c == d):
                            xymap = np.load(os.path.join(flow_dir,'{}_xymap_back.npy'.format(temp_string2))).astype(np.int32)
                            warped_closer_label = self.warp(closer_plabel,xymap)
                            warp_prob = self.warp(closer_prob,xymap)
                            #warp过去的像素至少是远景的置信度最大类或者第二大类
                            c = torch.argmax(prob,axis=-1)
                            prob_sort = torch.argsort(-1*prob,axis=-1).cpu()
                            pred = prob_sort[:,:,0]
                            pred2 = prob_sort[:,:,1]
                            mask3 = (warped_closer_label == pred) | (warped_closer_label == pred2)

                            #255表示未被划分为伪标签，20表示非可信光流的warp区域
                            mask = (warped_closer_label != 255) & (warped_closer_label != 20 ) & (label_f == 255) & mask3
                            concat_label = label_f.clone()


                            concat_label[mask] = warped_closer_label[mask]
                            #可信且已经分配伪标签的区域
                            mask2 = (label_f != 255) & (warped_closer_label != 20 ) & mask3
                            if torch.sum(mask2) > 0:
                                mean_prob = torch.zeros_like(prob)
                                mean_prob = (warp_prob+prob)/2
                                a =  torch.max(mean_prob[mask2],dim=1)[1]
                                a = a.type_as(concat_label)
                                concat_label[mask2] = a
                            concat_label = concat_label.cpu().numpy()
                            closer_plabel = closer_plabel.cpu().numpy()
                            warped_closer_label = warped_closer_label.cpu().numpy()
                            warped_closer_label[warped_closer_label == 20] = 255
                            cc +=1
                            Image.fromarray(Cityscapes.train_id_to_id[concat_label].astype(np.uint8)).save('%s/%s.png' % (self.save_concat_label_path,temp_string))
                            if args.save_val_results:
                                # d = concat_label[concat_label != label_f_n]
                                # a = np.sum((d == 6) | (d == 7) | (d == 12)  | (d == 14)
                                #   | (d == 15)  | (d == 16)  | (d == 17)  | (d == 18))
                                # # print(np.unique(d))
                                # if a < 2000:
                                #     continue
                                concat_label_d = Cityscapes.decode_target(concat_label.copy()).astype(np.uint8)
                                closer_plabel_d = Cityscapes.decode_target(closer_plabel.copy()).astype(np.uint8)
                                warped_closer_label_d = Cityscapes.decode_target(warped_closer_label.copy()).astype(np.uint8)
                                label_f_d = Cityscapes.decode_target(label_f_n.copy()).astype(np.uint8)
                                dark_mask = np.zeros_like(concat_label_d,dtype=np.uint8)
                                dark_mask[concat_label != label_f_n] = (255,140,0)
                                blend_pred = cv2.addWeighted(concat_label_d,0.5,dark_mask,0.5,0.0)
                                img_perceptual = cv2.hconcat([closer_plabel_d,warped_closer_label_d,label_f_d,concat_label_d,blend_pred])  # 水平拼接
                                Image.fromarray(img_perceptual.astype(np.uint8)).save('%s/%s.png' % (self.save_concat_label_color_path,temp_string))
                                Image.fromarray(concat_label_d.astype(np.uint8)).save('%s/%s_concat.png' % (self.save_concat_label_color_path,temp_string))
                        else:
                            cc += 1
                            Image.fromarray(Cityscapes.train_id_to_id[label_f_n].astype(np.uint8)).save('%s/%s.png' % (self.save_concat_label_path,temp_string))

                    closer_prob = prob
                    closer_pred = pred
                    closer_plabel = label_f
                    closer_img_name = sample_name
                    closer_img_path = sample_names[j]
                    # closer_img = image_d                            
        print('done')
        # print('cc:{}'.format(cc))
    def warp(self,closer_label,xymap):
        avalid_image = torch.zeros_like(closer_label,dtype=closer_label.dtype) + 20
        #warp
        avalid_mask = xymap[:,:,0]!=-1
        avalid_x,avalid_y = np.where(avalid_mask)  
        c = xymap[avalid_x,avalid_y,:]
        avalid_image[avalid_x,avalid_y] = closer_label[c[:,1],c[:,0]]
        return avalid_image

    def label_propagation_ov(self,model,device,loader,dense_flow_path,thresh_hold_path,args):
        print('###### Start label propagation in round {} ! ######'.format(self.round_idx))
        start_pl = time.time()
        cls_thresh = np.load(thresh_hold_path)
        ## output of deeplab is logits, not probability
        softmax2d = torch.nn.Softmax2d()
        with torch.no_grad():
            #遍历顺序为从近景到远景
            last_img_path,last_img_name,last_img,last_pred,last_prob = None,None,None,None,None
            prob,pred,last_plabel,proccessed_pred = None,None,None,None
            for i, datas in enumerate(tqdm(loader)):
                if(len(datas)==3):
                    images,_,sample_names = datas
                elif(len(datas) == 2):
                    images,_ = datas
                else:
                    print("output shape from dataloader is not correct")
                    return
                images = images.to(device, dtype=torch.float32)
                outputs,_ = model(images)

                if self.interp is None:
                    self.interp = torch.nn.Upsample(size=images.shape[2:], mode='bilinear')
                outputs_interpolated = softmax2d(self.interp(outputs)).cpu().numpy()
                for j in range(len(outputs_interpolated)):
                    pred_prob = outputs_interpolated[j]
                    sample_base_name = os.path.basename(sample_names[j])
                    sample_name = sample_base_name[:sample_base_name.rfind('.')]
                    image = images[j]
                    image = image.detach().cpu().numpy()
                    image_d  = (self.denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    
                    weighted_prob = pred_prob.transpose(1,2,0)/cls_thresh
                    weighted_prob = weighted_prob.transpose(2,0,1)
                    prob = weighted_prob
                    weighted_pred_trainIDs = np.asarray(np.argmax(weighted_prob, axis=0), dtype=np.uint8)

                    weighted_conf = np.amax(weighted_prob, axis=0)
                    pred_label_trainIDs = weighted_pred_trainIDs.copy()
                    pred_label_labelIDs = Cityscapes.train_id_to_id[pred_label_trainIDs]
                    pred_label_labelIDs[weighted_conf < 1] = 0  # '0' in cityscapes indicates 'unlabaled' for labelIDs
                    pred_label_trainIDs[weighted_conf < 1] = 255 # '255' in cityscapes indicates 'unlabaled' for trainIDs
                    

                    #generate plabel
                    plabel = pred_label_trainIDs.copy()
                    if(last_img_path is not None):
                        temp_string = sample_name
                        far_img_id = temp_string[-6:]
                        temp_string2 = os.path.basename(last_img_path)
                        temp_string2 = temp_string2.split('.png')[0]
                        closer_img_id = temp_string2[-6:]
                        a = int(far_img_id)
                        b = int(closer_img_id)
                        c = os.path.basename(temp_string)[:10]
                        d = os.path.basename(last_img_path)[:10] 
                        if(a == b - 1 and c == d):
                            #loding dense coresponding
                            flow = np.load(os.path.join(dense_flow_path,temp_string + '_2_' + temp_string2+'.npy'))
                            proccessed_pred,proccessed_conf,proccessed_prob = self.refineSeg(last_img,image_d,last_prob,prob,flow,
                            plabel,last_plabel,save_color_result_path = '%s/%s_color.png' % (self.save_proccess_label_color_path,sample_name))

                    if proccessed_pred is not None:
                        pred_label_trainIDs = proccessed_pred
                        pred_label_labelIDs = Cityscapes.train_id_to_id[proccessed_pred]
                    # #add position prior
                    # prior_error_mask = None
                    # if args.useprior:
                    #     prior_error_mask = np.zeros_like(pred_label_trainIDs,dtype=bool)
                    #     for k,pmap in self.prior_map.items():
                    #         a = pred_label_trainIDs == k
                    #         b = pmap < 25
                    #         c = a & b
                    #         prior_error_mask[c] = True
                    #     pred_label_trainIDs[prior_error_mask] = 255
                    #     pred_label_labelIDs[prior_error_mask] = 0
                    
                    # pseudo-labels with labelID
                    pseudo_label_labelIDs = pred_label_labelIDs.copy()
                    pseudo_label_trainIDs = pred_label_trainIDs.copy()
                    # if(args.save_val_results):
                    #     wpred_label_col = Cityscapes.decode_target(pseudo_label_trainIDs).astype(np.uint8)
                    #     Image.fromarray(wpred_label_col).save('%s/%s_color.png' % (self.save_proccess_label_color_path,sample_name))
                    Image.fromarray(pseudo_label_labelIDs.astype(np.uint8)).save('%s/%s.png' % (self.save_proccess_label_path,sample_name))
                    
                    last_prob = prob
                    last_pred = pred
                    last_plabel = plabel
                    last_img_name = sample_name
                    last_img_path = sample_names[j]
                    last_img = image_d

        print('###### Finish label propagation in round {}! Time cost: {:.2f} seconds. ######'.format(self.round_idx,time.time() - start_pl))
        return 0
    
    def concat_label(self,save_propagation_label_path,save_superpixel_extend_label_path):
        plabel_se_list = os.listdir(save_superpixel_extend_label_path)
        for plabel_name in tqdm(plabel_se_list):
            if (plabel_name.find('_change') > 0):
                continue
            sample_name = plabel_name.split('.png')[0]
            file_path = osp.join(save_superpixel_extend_label_path,plabel_name)
            file2_path = osp.join(save_propagation_label_path,plabel_name)
            plabel_se = np.array(Image.open(file_path))
            plabel_p = np.array(Image.open(file2_path))
            # mask_p = plabel_p != 255
            # plabel_se = plabel_se != 255
            concat_label = plabel_p.copy()
            concat_label[plabel_se !=0] = plabel_se[plabel_se != 0]
            Image.fromarray(concat_label.astype(np.uint8)).save('%s/%s.png' % (self.save_concat_label_path,sample_name))
            concat_label_d = Cityscapes.decode_target(Cityscapes.encode_target(concat_label.copy())).astype(np.uint8)
            plabel_p_d = Cityscapes.decode_target(Cityscapes.encode_target(plabel_p.copy())).astype(np.uint8)
            plabel_se_d = Cityscapes.decode_target(Cityscapes.encode_target(plabel_se.copy())).astype(np.uint8)
            if np.sum(concat_label != plabel_se) > 0:
                dark_mask = np.zeros_like(concat_label_d,dtype=np.uint8)
                dark_mask[concat_label != plabel_se] = (255,140,0)
                blend_pred = cv2.addWeighted(concat_label_d,0.5,dark_mask,0.5,0.0)
                img_perceptual = cv2.hconcat([plabel_p_d,plabel_se_d,concat_label_d,blend_pred])  # 水平拼接
                Image.fromarray(img_perceptual.astype(np.uint8)).save('%s/%s.png' % (self.save_concat_label_color_path,sample_name))
                # Image.fromarray(concat_label_d.astype(np.uint8)).save('%s/%s.png' % (self.save_concat_label_color_path,sample_name))
                # Image.fromarray(plabel_p_d.astype(np.uint8)).save('%s/%s_p.png' % (self.save_concat_label_color_path,sample_name))
    
    def warp_and_concat_m(self,pseudolabel_path,xymap_dir,save_results,resize_w = 720,resize_h = 1280):
        print('###### Start label propagation in round {} ! ######'.format(self.round_idx))
        #thread_nums = 4
        threads_pools = []
        imgs_list_1 = glob.glob(os.path.join(pseudolabel_path,'1508039851*.png'))
        imgs_list_2 = glob.glob(os.path.join(pseudolabel_path,'1508040913*.png'))
        imgs_list_3 = glob.glob(os.path.join(pseudolabel_path,'1508041975*.png'))
        imgs_list_4 = glob.glob(os.path.join(pseudolabel_path,'1508043037*.png'))
        print('label propagation len:{}'.format(len(imgs_list_1)+len(imgs_list_2)+len(imgs_list_3)+len(imgs_list_4)))
        imgs_list_all = [imgs_list_1,imgs_list_2,imgs_list_3,imgs_list_4]
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(len(imgs_list_all)) # 7.7G
        for thread_id,img_list in enumerate(imgs_list_all):
            if thread_id == 0:
                continue
            pool.apply_async(self.warp_and_concat, args=(pseudolabel_path,xymap_dir,save_results,resize_w,resize_h,
            img_list,thread_id))
        self.warp_and_concat(pseudolabel_path,xymap_dir,save_results,resize_w,resize_h,
        imgs_list_all[0],0)
        pool.close()
        pool.join()
        print('###### end label propagation in round {} ! ######'.format(self.round_idx))

    def warp_and_concat(self,pseudolabel_path,xymap_dir,save_results,resize_w = 720,resize_h = 1280,imgs_list=None,thread_id=0):
        print('###### Start flow information generation ######')
        start_pl = time.time()
        ## output of deeplab is logits, not probability
        if imgs_list is None:
            imgs_list = glob.glob(os.path.join(pseudolabel_path,'*.png'))
        imgs_list_sorted = natsorted(imgs_list)
        print('imgs_list_sorted:{} '.format(len(imgs_list_sorted)))
        for i,closer_img_path in enumerate(imgs_list_sorted):
            if thread_id == 0 and  i % 100 == 0:
                print("done {} iteration, total:{}".format(i,len(imgs_list_sorted)))
            far_img = imgs_list_sorted[i-1]
            temp_string = os.path.basename(far_img)
            temp_string = temp_string.split('.png')[0]
            far_img_id = temp_string[-6:]
            temp_string2 = os.path.basename(closer_img_path)
            temp_string2 = temp_string2.split('.png')[0]
            closer_img_id = temp_string2[-6:]
            a = int(far_img_id)
            b = int(closer_img_id)
            c = os.path.basename(far_img)[:10]
            d = os.path.basename(closer_img_path)[:10] 
            img_closer = cv2.imread(closer_img_path,0)
            if(a == b - 1 and c == d and i != 0):
                xymap = np.load(os.path.join(xymap_dir,'{}_xymap.npy'.format(temp_string2))).astype(np.int32)
                img_closer_r = cv2.resize(img_closer,(resize_h,resize_w),interpolation=cv2.INTER_NEAREST)
                img_far = cv2.imread(far_img,0)
                avalid_image = np.zeros_like(img_closer_r,dtype=np.uint8)
                avalid_x,avalid_y = np.where((xymap[:,:,0] !=-1) & (img_closer_r != 0))
                for i in range(len(avalid_x)):
                    avalid_image[xymap[avalid_x[i],avalid_y[i]][0],xymap[avalid_x[i],avalid_y[i]][1]] = img_closer_r[avalid_x[i],avalid_y[i]]
                avalid_image = cv2.resize(avalid_image,img_closer.shape[::-1],interpolation=cv2.INTER_NEAREST)
                mask = (avalid_image != 0) & (img_far == 0)
                concat_label = img_far.copy()
                concat_label[mask] = avalid_image[mask]
                Image.fromarray(concat_label.astype(np.uint8)).save('%s/%s.png' % (self.save_concat_label_path,temp_string))
                if save_results:
                    c = Cityscapes.encode_target(concat_label.copy())
                    a = np.sum((concat_label != c) & (c != 0) & (c != 1) & (c != 2) & (c != 10) & (c != 5) & (c != 0))
                    if a < 500:
                        continue
                    wrap_p_d = Cityscapes.decode_target(Cityscapes.encode_target(avalid_image.copy())).astype(np.uint8)
                    img_closer_d = Cityscapes.decode_target(Cityscapes.encode_target(img_closer.copy())).astype(np.uint8)
                    img_far_d = Cityscapes.decode_target(Cityscapes.encode_target(img_far.copy())).astype(np.uint8) 
                    concat_label_d = Cityscapes.decode_target(Cityscapes.encode_target(concat_label.copy())).astype(np.uint8)
                    dark_mask = np.zeros_like(concat_label_d,dtype=np.uint8)
                    dark_mask[concat_label != img_far] = (255,140,0)
                    blend_pred = cv2.addWeighted(concat_label_d,0.5,dark_mask,0.5,0.0)
                    img_perceptual = cv2.hconcat([img_closer_d,wrap_p_d,img_far_d,concat_label_d,blend_pred])  # 水平拼接
                    Image.fromarray(img_perceptual.astype(np.uint8)).save('%s/%s.png' % (self.save_concat_label_color_path,temp_string))
            else:
                Image.fromarray(img_closer.astype(np.uint8)).save('%s/%s.png' % (self.save_concat_label_path,temp_string))

# def generate_flow_infor(imgs_path,GLUNet,warp,device,save_path,save_warp_path):
#     print('###### Start flow information generation ######')
#     start_pl = time.time()
#     ## output of deeplab is logits, not probability
#     imgs_list = glob.glob(os.path.join(imgs_path,'*.png'))
#     imgs_list_sorted = natsorted(imgs_list)
#     print(imgs_list_sorted[:20])
#     with torch.no_grad():
#         for i,closer_img_path in enumerate(tqdm(imgs_list_sorted)):
#             far_img = imgs_list_sorted[i-1]
#             if i == 0:
#                 continue
#             temp_string = os.path.basename(far_img)
#             temp_string = temp_string.split('.png')[0]
#             far_img_id = temp_string[-6:]
#             temp_string2 = os.path.basename(closer_img_path)
#             temp_string2 = temp_string2.split('.png')[0]
#             closer_img_id = temp_string2[-6:]
#             a = int(far_img_id)
#             b = int(closer_img_id)
#             c = os.path.basename(far_img)[:10]
#             d = os.path.basename(closer_img_path)[:10] 
#             # print('in')
#             if(a == b - 1 and c == d):
#                 warped_source_image,_,flow = warp(closer_img_path,far_img,GLUNet,device)
#                 img_closer = np.array(Image.open(closer_img_path).convert('RGB'))
#                 img_far = np.array(Image.open(far_img).convert('RGB'))
#                 img_perceptual = cv2.hconcat([img_closer,img_far, warped_source_image])  # 水平拼接
#                 Image.fromarray(img_perceptual).save('%s/%s_preceptual.png' % (save_warp_path,os.path.basename(closer_img_path).split('.png')[0]))
#                 np.save(os.path.join(save_path, temp_string + '_2_' + temp_string2), flow)
#                 Image.fromarray(warped_source_image).save('%s/%s_warp_img.png' % (save_warp_path,os.path.basename(closer_img_path).split('.png')[0]))
#     print('###### Finish flow information generation! Time cost: {:.2f} seconds. ######'.format(time.time() - start_pl))

    

# if __name__ == '__main__':
#     from GLUnet.test import create_network,warp
#     GLUNet = create_network()
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # dt.misc.set_seed(4)
#     if(str(device)=='cuda'):
#         torch.backends.cudnn.deterministic = True
#     print("Device: %s" % device)
#     imgs_path = 'datasets/targetImgs'
#     save_path = 'results/flows'
#     save_warp_path = 'results/warp'
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#     if not os.path.exists(save_warp_path):
#         os.mkdir(save_warp_path)
#     generate_flow_infor(imgs_path,device=device,save_path=save_path,save_warp_path=save_warp_path,GLUNet=GLUNet,warp=warp)