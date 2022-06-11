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
Date: 2021-06-29 09:19:51
Description: file content
'''

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import grad

from datasets import Cityscapes
import cv2 as cv2
import matplotlib.pyplot as plt

import utils
class MomentumNet(nn.Module):

    def __init__(self, baseNet, slow_copy, args):
        super(MomentumNet, self).__init__()

        self.baseNet = baseNet
        self.momentum = args.net_momentum
        # initialising slow net
        self.teacherNet = slow_copy
        self.teacherNet.eval()
        self.class_numbers = args.num_classes
        for p in self.teacherNet.parameters():
            p.requires_grad = False
        self.register_buffer("prototypes", torch.zeros(args.num_classes,args.low_dim))

        # debug log
        self.corect_num = 0
        self.refine_num = 0
        self.corect_2_error_num = 0
        # 第x类的精确率
        self.change_correct_mask = np.zeros((args.num_classes, args.num_classes))
        self.change_sum_mask = np.zeros((args.num_classes, args.num_classes))
        # 置信度排名中的第x类转为第1类的精确率，不包括没有变化的像素以及gt中255的像素
        self.change_correct_mask2 = np.zeros((args.num_classes))
        # 记录从置信度排名中第i类转为第一大类的像素总数，不包括没有变化的像素以及gt中255的像素
        self.change_sum_mask2 = np.zeros((args.num_classes))

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update"""
        # parameter
        for param_q, param_k in zip(self.baseNet.parameters(), self.teacherNet.parameters()):
            param_k.data = param_k.data.clone() * self.momentum + param_q.data.clone() * (1. - self.momentum)

    def process_label(self, label):
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, self.class_numbers + 1, w, h).cuda()
        id = torch.where(label < self.class_numbers, label, torch.Tensor([self.class_numbers]).cuda())
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1


    def calculate_mean_vector(self, feat_cls, outputs, plabel = None, labels_val=None, model=None):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        conf = outputs_softmax.max(dim=1, keepdim=True)[0]
        mask = torch.ones_like(conf, dtype=bool)
        if plabel is not None:
            mask[plabel == 255] = False

        outputs_argmax = self.process_label(outputs_argmax.float())
        if labels_val is None:
            outputs_pred = outputs_argmax
        else:
            labels_expanded = self.process_label(labels_val)
            outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred * mask, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t] * mask[n]
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def forward(self, x, \
                      use_teacher=False,update_teacher=False, plabel = None, \
                       update_prototype = False, refine_outputs = False, labels =  None):
        """Args:
                x: input images [BxCxHxW]
                y: ground-truth for source images [BxHxW]
                x2: input images w/o photometric noise [BxCxHxW]
                T: length of the sequences
        """
    
        if update_teacher:
            # print("Updating the teacher")
            # teacher_stu_losses = self._momentum_update(True)
            self._momentum_update()

        if use_teacher:
            res = self.teacherNet(x)
        else:
            res = self.baseNet(x)

        outputs,feats = res

        # outputs2 = torch.nn.functional.interpolate(
        #     outputs, size=feats.shape[2:], mode="bilinear", align_corners=False
        # )
        # preds = torch.argmax(outputs2, dim=1)
        # self._enqueue(labels=preds, feats=feats)

        if update_prototype:
            # 更新原型
            plabel = plabel.unsqueeze(1).float().clone()
            plabel = torch.nn.functional.interpolate(plabel,
                                                    (feats.shape[2], feats.shape[3]), mode='nearest')
            vectors, ids = self.calculate_mean_vector(feats, outputs, plabel)
            #vectors, ids = class_features.calculate_mean_vector_by_output(feat_cls, output, model)
            for t in range(len(ids)):
                vector, id = vectors[t].detach(), ids[t]
                self.prototypes[id] = self.prototypes[id] * self.momentum +  (1 - self.momentum) * vector.squeeze()


        # 计算相似度
        if refine_outputs:
            predicetd_scores = torch.softmax(outputs, dim=1)
            max_scores, pseudo_labels = torch.max(predicetd_scores, dim=1)
            # using partial labels to filter out negative labels

            # compute protoypical logits
            prototypes = self.prototypes.clone().detach()
            
            logits_prot = torch.matmul(feats.detach().permute(0,2,3,1), prototypes.t())
            score_prot = torch.softmax(logits_prot, dim=1)
            score_prot = score_prot.permute(0,3,1,2)
            # max_scores2, pseudo_labels2 = torch.max(score_prot, dim=1)

            #置信度最大类或者第二三大类
            prob_sort = torch.argsort(-1*outputs,dim=1)
            prob_sort = prob_sort.permute(1,0,2,3)
            pred1 = prob_sort[0]
            pred2 = prob_sort[1]
            pred3 = prob_sort[2]
            logits_mask1 = torch.zeros_like(prob_sort, dtype=bool)
            logits_mask1 = logits_mask1.scatter_(0,pred1.unsqueeze(dim=0),1)
            # _, ttt = torch.max(logits_mask1, dim=1)
            logits_mask1 = logits_mask1.scatter_(0,pred2.unsqueeze(dim=0),1)
            logits_mask1 = logits_mask1.scatter_(0,pred3.unsqueeze(dim=0),1)
            logits_mask1 = logits_mask1.permute(1,0,2,3)
            # cc = torch.sum(logits_mask1, dim=1)

            # 按照mask 重新分配输出
            ttt = outputs * logits_mask1
            _sum = torch.sum(ttt, dim=1).unsqueeze(dim=1)

            ttt2 = score_prot * logits_mask1
            _sum2 = torch.sum(ttt2, dim=1).unsqueeze(dim=1)
            ttt2 /= _sum2
            # kkk = torch.sum(ttt2, dim=1)
            ttt2 *= _sum

            outputs2 = torch.nn.functional.interpolate(
                outputs, size=feats.shape[2:], mode="bilinear", align_corners=False
            )

            preds_before_process = torch.argmax(outputs2, dim=1).detach().cpu().numpy()
            # outputs[logits_mask1] = 0.5 * outputs[logits_mask1] + 0.5 * ttt2[logits_mask1]
            # mask2 = (plabel == 255).repeat(1, outputs.shape[1], 1, 1)
            # outputs[mask2] = (outputs * ( 0.5 + 0.5 * score_prot))[mask2]
            outputs = outputs * ( 0.5 + 0.5 * score_prot)
            # outputs *= score_prot

            outputs2 = torch.nn.functional.interpolate(
                outputs, size=feats.shape[2:], mode="bilinear", align_corners=False
            )
            preds_after_process = torch.argmax(outputs2, dim=1).detach().cpu().numpy()
        

            if labels is not None:
                labels = F.interpolate(labels.unsqueeze(1).float(), size=feats.shape[2:], mode='nearest').cpu().numpy()
                labels = labels.squeeze(axis = 1).astype(np.int64)
            
            # 统计置信度第j类转为第1大类的正确率
            torch_preds_after_process = torch.argmax(outputs2, dim=1).detach()
            c = prob_sort == torch_preds_after_process.unsqueeze(dim=0)
            aaa = ((prob_sort + 1) * c).cpu().numpy()
            change_conf_marks = np.argmax(aaa, axis = 0)

            plabel_cp = plabel.cpu().int().numpy()
            for i in range(len(preds_before_process)):
                change_conf_mark = change_conf_marks[i]
                pred_before_process = preds_before_process[i]
                pred_after_process = preds_after_process[i]
                mask = pred_before_process != pred_after_process
                label = None
                if labels is not None:
                    label = labels[i]
                    mask2 = mask & (label != 255)
                    # mask3 = mask2 & (plabel_cp[i][0] == 255)
                    self.refine_num += np.sum(mask2)
                    # 修改分母为整张图片的像素点
                    # self.refine_num += mask2.shape[0] * mask2.shape[1]
                    self.corect_num += np.sum(pred_after_process[mask2] == label[mask2])
                    self.corect_2_error_num += np.sum(mask2)
                    # self.corect_2_error_num += np.sum(pred_before_process[mask2] == label[mask2])
                    # 统计发生修改的像素点比例
                    # self.corect_2_error_num += np.sum(mask3)
                
            #     # 统计第j类转为第k类的正确率
            #     for j in range(self.class_numbers):
            #         for k in range(self.class_numbers):
            #             mask_change_j = mask & (pred_before_process == j) & \
            #                  (pred_after_process == k) & (label == k)
            #             self.change_correct_mask[j][k] += np.sum(mask_change_j)
            #             mask_change_total = mask & (pred_before_process == j) & \
            #                  (pred_after_process == k) & (label != 255)
            #             self.change_sum_mask[j][k] += np.sum(mask_change_total)

            #     for kk in range(self.class_numbers):
            #         mask_conf = change_conf_mark == kk
            #         mask_conf &= mask
            #         mask_conf &= (label != 255)
            #         self.change_sum_mask2[kk] += np.sum(mask_conf)
            #         mask_conf &= (label == pred_after_process)
            #         self.change_correct_mask2[kk] += np.sum(mask_conf)

                # # a = np.sum(mask);
                # if(np.sum((mask & (label != 255))) < 200): continue
                # pred_after_process_d = Cityscapes.decode_target(pred_after_process).astype(np.uint8)
                # pred_before_process_d = Cityscapes.decode_target(pred_before_process).astype(np.uint8)
                # label_d = Cityscapes.decode_target(label).astype(np.uint8)
                # pred_before_process_d = Image.fromarray(pred_before_process_d)
                # pred_after_process_d = Image.fromarray(pred_after_process_d)
                # label_d = Image.fromarray(label_d)
                # dark_mask = np.zeros_like(pred_after_process_d,dtype=np.uint8)
                # dark_mask[mask] = (255,140,0)

                # image2 = torch.nn.functional.interpolate(
                #     x, size=feats.shape[2:], mode="bilinear", align_corners=False
                # )    

                # image = image2[i].detach().cpu().numpy()
                # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                #                 std=[0.229, 0.224, 0.225])
                # image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                # image = Image.fromarray(image)
                # plt.figure()
    
                # plt.subplot(221)
                # plt.imshow(pred_before_process_d)
                
                # plt.subplot(222)
                # plt.imshow(pred_after_process_d)
                
                # plt.subplot(223)
                # # 图片格式调整一致
                # image = image.convert('RGBA')
                # label_d = label_d.convert('RGBA')
                # image = Image.blend(image, label_d, 0.4)
                # plt.imshow(image)

                # plt.subplot(224)
                # plt.imshow(dark_mask)
                
                # plt.savefig('test.png',bbox_inches='tight')
        t = np.sum(self.change_sum_mask)
        return outputs,feats
