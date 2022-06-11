from torch.autograd.variable import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np
import utils
from PIL import Image
import cv2 as cv2
class ConstractLoss(nn.Module):
    def __init__(self):
        super(ConstractLoss, self).__init__()
        self.temperature=0.07
        self.base_temperature=0.07
        self.max_sample = 19
        self.feats_pairs = None

    def recover(self, image_c, image_f, xymap, warp_params,num_class = 19):
        off_i,off_j,isFliped = warp_params
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        xymap = xymap.numpy()
        num_bs = len(image_c)
        #保存近景(1080x1920)中对应的点的特征，2表示同一类别的只有2个特征点，一个在近景，一个远景，这个函数内只赋值了近景
        results = torch.zeros((num_bs, num_class, 2, image_c.size()[-1]), dtype=torch.float).cuda()
        results_xy = np.zeros((num_bs,num_class,2)) #保存远景(768x768)中对应的点的坐标位置
        for bs in range(num_bs):
            image_c_bs = image_c[bs].cpu().detach().numpy()
            image_f_bs = image_f[bs].cpu().detach().numpy()
            image_c_bs = (denorm(image_c_bs) * 255).transpose(1, 2, 0).astype(np.uint8)
            image_f_bs = (denorm(image_f_bs) * 255).transpose(1, 2, 0).astype(np.uint8)
            avalid_image = np.zeros_like(image_f_bs)
            xymap_bs = xymap[bs]
            isFliped_bs = isFliped[bs]
            #warp
            avalid_mask = xymap_bs[:,:,0]!=-1
            c = 0
            avalid_x,avalid_y = np.where(avalid_mask)  
            for i in range(len(avalid_x)):
                y,x = xymap_bs[avalid_x[i],avalid_y[i],:] #xymap[:,:,0]对应近景的某列,xymap[:,:,1]对应行
            if isFliped_bs:
                avalid_image[avalid_x[i], avalid_image.shape[0] - 1 - avalid_y[i]] = image_c_bs[x,y]
            else:
                avalid_image[avalid_x[i], avalid_y[i]] = image_c_bs[x,y]

            Image.fromarray(image_c_bs).save('image.png' )
            Image.fromarray(avalid_image.astype(np.uint8)).save('image2.png')
            Image.fromarray(image_f_bs.astype(np.uint8)).save('image3.png')

    # def getCloserFeats(self, feats_c, label_f, xymap, warp_params,num_class = 19):
    #     off_i,off_j,isFliped = warp_params
    #     feats_c = feats_c.permute(0,2,3,1)
    #     num_bs = len(feats_c)
    #     #保存近景(1080x1920)中对应的点的特征，2表示同一类别的只有2个特征点，一个在近景，一个远景，这个函数内只赋值了近景
    #     results = torch.zeros((num_bs, num_class, 2, feats_c.size()[-1]), dtype=torch.float).cuda()
    #     results_xy = np.zeros((num_bs,num_class,2),dtype=np.int16) #保存远景(768x768)中对应的点的坐标位置
    #     ava_class_id = np.zeros((num_bs,num_class),dtype=bool)
    #     for bs in range(num_bs):
    #         xymap_bs = xymap[bs]
    #         isFliped_bs = isFliped[bs]
    #         label_f_bs = label_f[bs]
    #         feats_c_bs = feats_c[bs]
    #         #warp
    #         avalid_mask = xymap_bs[:,:,0]!=-1
    #         for class_id in range(num_class):
    #             avalid_x,avalid_y = np.where(avalid_mask & (label_f_bs == class_id))  
    #             l = len(avalid_x)
    #             if l > 0:
    #                 ava_class_id[bs,class_id] = True
    #                 i = np.random.randint(l)
    #                 y,x = xymap_bs[avalid_x[i],avalid_y[i],:] #xymap[:,:,0]对应近景的某列,xymap[:,:,1]对应行
    #                 results[bs,class_id,0] = feats_c_bs[x,y]
    #                 if isFliped_bs:
    #                     results_xy[bs,class_id] = avalid_x[i], label_f_bs.shape[0] - 1 - avalid_y[i]
    #                 else:
    #                     results_xy[bs,class_id] = avalid_x[i],avalid_y[i]
    #     return results,results_xy,ava_class_id

    def resize_pos(self,x1,y1,src_size,tar_size):
    
        w1=src_size[0]
        h1=src_size[1]
        w2=tar_size[0]
        h2=tar_size[1]
        y2=(h2/h1)*y1
        x2=(w2/w1)*x1
        return int(x2),int(y2)

    def getCloserFeats(self, feats_c, label_f, xymap, warp_params,num_class = 19):
        off_i,off_j,isFliped = warp_params
        feats_c = feats_c.permute(0,2,3,1)
        num_bs = len(feats_c)
        w,h = feats_c.size()[1:3]
        o_w,o_h = w*4,h*4
        label_o_w,label_o_h = label_f.size()[-2:]
        label_o_w,label_o_h = int(label_o_w / 4), int(label_o_h / 4)
        #保存近景(1080x1920)中对应的点的特征，2表示同一类别的只有2个特征点，一个在近景，一个远景，这个函数内只赋值了近景
        results = torch.zeros((num_bs, num_class, 2, feats_c.size()[-1]), dtype=torch.float).cuda()
        results_xy = np.zeros((num_bs,num_class,2),dtype=np.int16) #保存远景(768x768)中对应的点的坐标位置
        ava_class_id = np.zeros((num_bs,num_class),dtype=bool)
        for bs in range(num_bs):
            xymap_bs = xymap[bs]
            isFliped_bs = isFliped[bs]
            label_f_bs = label_f[bs]
            feats_c_bs = feats_c[bs]
            #warp
            c = 1
            avalid_mask = xymap_bs[:,:,0]!=-1
            for class_id in range(num_class):
                avalid_x,avalid_y = np.where(avalid_mask & (label_f_bs == class_id))  
                l = len(avalid_x)
                if l > 0:
                    ava_class_id[bs,class_id] = True
                    i = np.random.randint(l)
                    y,x = xymap_bs[avalid_x[i],avalid_y[i],:] #xymap[:,:,0]对应近景的某列,xymap[:,:,1]对应行
                    new_a_x,new_a_y = self.resize_pos(avalid_x[i],avalid_y[i],(label_f_bs.size()[:2]),(label_o_w,label_o_h))
                    new_x,new_y = self.resize_pos(x,y,(o_w,o_h),(w,h))
                    results[bs,class_id,0] = feats_c_bs[new_x,new_y]
                    if isFliped_bs:
                        results_xy[bs,class_id] = new_a_x, label_o_w - 1 - new_a_y
                    else:
                        results_xy[bs,class_id] = new_a_x, new_a_y
        return results,results_xy,ava_class_id

    def _contrastive(self, feats_pairs):
        anchor_num, n_view = feats_pairs.shape[0], feats_pairs.shape[1]

        labels_ = torch.arange(0, anchor_num).view(-1, 1)

        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_pairs, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    # def forward(self,feats_f,feats_pair,feats_pair_xy,ava_class_id):
    #     loss = 0
    #     feats_pair = feats_pair.cuda()
    #     for bs in range(feats_pair_xy.shape[0]):
    #         for class_id in range(feats_pair_xy.shape[1]):
    #             if ava_class_id[bs,class_id]:
    #                 x,y = feats_pair_xy[bs,class_id]
    #                 feats_pair[bs,class_id] = feats_f[bs,:,x,y]
    #             t_feats_pairs = feats_pair[bs,ava_class_id[bs]]
    #             if len(t_feats_pairs) == 0:
    #                 continue
    #             t_feats_pairs = F.normalize(t_feats_pairs, p=2, dim=2)
    #             if loss is 0:
    #                 loss = self._contrastive(t_feats_pairs)
    #             else:
    #                 loss += self._contrastive(t_feats_pairs) 
    #     return loss
         

    
    def forward(self, feats_c, feats_f, label_f, xymap, pred_c, warp_params):
        off_i,off_j,isFliped = warp_params
        loss = 0
        xymap = xymap.detach().cpu().numpy()
        label_f = label_f.detach().cpu().numpy()
        w,h = feats_c.size()[1:3]
        o_w,o_h = w*4,h*4
        label_o_w,label_o_h = label_f.shape[-2:]
        label_o_w,label_o_h = int(label_o_w / 4), int(label_o_h / 4)
        feats_c, feats_f = feats_c.permute(0,2,3,1), feats_f.permute(0,2,3,1)
        for bs in range(len(feats_c)):
            feats_pairs = None
            xymap_bs = xymap[bs]
            isFliped_bs = isFliped[bs]
            feats_c_bs = feats_c[bs]
            pred_c_bs = pred_c[bs]
            feats_f_bs = feats_f[bs]
            label_f_bs = label_f[bs]
            #warp
            avalid_mask = xymap_bs[:,:,0]!=-1
            c = 0
            pick_class_id = []
            for class_id in range(19):
                avalid_x,avalid_y = np.where(avalid_mask & (label_f_bs == class_id))
                l = len(avalid_x)
                feats_pair,new_x,new_y,new_a_x,new_a_y,f_s = None,None,None,None,None,None
                f_c = None
                if l > 0:
                    c_in = 0
                    p_c = -1
                    while(p_c != class_id and c_in < 5):
                        c_in += 1
                        i = np.random.randint(l)
                        y,x = xymap_bs[avalid_x[i],avalid_y[i],:] #xymap[:,:,0]对应近景的某列,xymap[:,:,1]对应行
                        new_a_x,new_a_y = self.resize_pos(avalid_x[i],avalid_y[i],(label_f_bs.shape[:2]),(label_o_w,label_o_h))
                        new_x,new_y = self.resize_pos(x,y,(o_w,o_h),(w,h))
                        f_c = feats_c_bs[new_x,new_y].unsqueeze(dim=0)
                        if isFliped_bs:
                            f_s = feats_f_bs[new_a_x,feats_f_bs.shape[0] - 1 - new_a_y]
                            p_c = pred_c_bs[new_x,feats_f_bs.shape[0] - 1 - new_y]
                        else:
                            f_s = feats_f_bs[new_a_x,new_a_y]
                            p_c = pred_c_bs[new_x,new_y]
                        f_s = f_s.unsqueeze(dim=0)
                    if c_in == 5:
                        continue
                    pick_class_id.append((p_c,class_id))
                    feats_pair = torch.cat((f_c,f_s),dim=0)
                    if feats_pairs is None:
                        feats_pairs = feats_pair.unsqueeze(dim=0)
                    else:
                        feats_pairs = torch.cat((feats_pairs,feats_pair.unsqueeze(dim=0)),dim=0)
                    c+=1
            if c == 0:
                continue
            feats_pairs = F.normalize(feats_pairs, p=2, dim=2)
            if loss is 0:
                loss = self._contrastive(feats_pairs)
            else:
                loss += self._contrastive(feats_pairs)
            del feats_pairs
        return loss / len(feats_c)
            


