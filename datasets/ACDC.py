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
Date: 2022-04-30 01:25:58
Description: file content
'''

import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import copy
import glob
from utils import ext_transforms as et

class ACDC(data.Dataset):
    """
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/ACDCScripts
    ACDCClass = namedtuple('ACDCClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        ACDCClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        ACDCClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        ACDCClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        ACDCClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        ACDCClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        ACDCClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        ACDCClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        ACDCClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        ACDCClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        ACDCClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        ACDCClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        ACDCClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        ACDCClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        ACDCClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        ACDCClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        ACDCClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        ACDCClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        ACDCClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        ACDCClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        ACDCClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        ACDCClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        ACDCClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        ACDCClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        ACDCClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        ACDCClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        ACDCClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        ACDCClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        ACDCClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        ACDCClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        ACDCClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        ACDCClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        ACDCClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        ACDCClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        ACDCClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        ACDCClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, mode = 'fine', train_type = 'test' , condiction_type = 'fog', transform=None,\
         pseudo_label_dir = None, clean_plabel_dir = None, return_transform_p = False, need_clean = False,\
         soft_label_dir = None
             ):
        self.root = os.path.expanduser(root)
        self.target_type = 'labelIds'

        self.test_only = True if train_type == 'test' else False
        self.pseudo_label_dir = pseudo_label_dir
        self.transform = transform
        self.return_transform_p = return_transform_p
        self.images = []
        self.targets = []
        # 高置信度的伪标签
        self.conf_plabels = []
        # debug
        self.gts = []
        self.soft_labels = []
        self.soft_labels_init = []
        self.record_softlb = {}
        self.need_clean = need_clean
        self.clean_plabel_dir = clean_plabel_dir
        self.clean_plabels_path = []
        self.img_size = (1920, 1080)
        self.images_dir = os.path.join(self.root, 'rgb_anon', condiction_type, train_type)
        # 白天的图像
        # self.images_dir = os.path.join(self.root, 'rgb_anon', condiction_type, train_type+'_ref')
        self.targets_dir = os.path.join(self.root, 'gt', condiction_type, train_type)

        for scence in os.listdir(self.images_dir):
            imagelist = glob.glob(os.path.join(self.images_dir, scence, '*.png'))
            for imagepath in imagelist:
                file_name = os.path.basename(imagepath)
                # if file_name.find('GOPR0476_frame_000781') < 0 and file_name.find('GP010476_frame_000022') < 0 \
                # and file_name.find('GP020475_frame_000081') < 0:
                #     continue
                clean_file_name = file_name.replace('rgb_anon','rgb_ref_anon')
                # file_name = file_name.replace('rgb_ref_anon','gt_labelIds')
                file_name = file_name.replace('rgb_anon','gt_labelIds')
                self.images.append(imagepath)
                if self.pseudo_label_dir is not None:
                    self.targets.append(os.path.join(self.pseudo_label_dir, os.path.basename(imagepath)))
                    t = self.pseudo_label_dir[:self.pseudo_label_dir.rfind('/')]
                    self.conf_plabels.append(os.path.join(t, 'pseudo_label_weighted', os.path.basename(imagepath)))
                    self.soft_labels_init.append(os.path.join(t, 'soft_labels', os.path.basename(imagepath).replace('png','npy')))
                    if not os.path.exists(os.path.join(t, 'soft_labels_init')):
                        os.mkdir(os.path.join(t, 'soft_labels_init'))
                    self.soft_labels.append(os.path.join(t, 'soft_labels_init', os.path.basename(imagepath).replace('png','npy')))
                    if self.clean_plabel_dir is not None:
                        # self.clean_plabels_path.append(os.path.join(self.clean_plabel_dir, os.path.basename(clean_file_name)))
                        self.clean_plabels_path.append(os.path.join(self.clean_plabel_dir, os.path.basename(imagepath)))
                    # debug
                    self.gts.append(os.path.join(self.targets_dir, scence, file_name))
                else:   
                    self.targets.append(os.path.join(self.targets_dir, scence, file_name))
        # self.images = self.images[:10]
        # self.targets = self.targets[:10]
        
        # print(self.images[:10])
        # print(self.targets[:10])

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]


    def update_soft_label(self, idx, new_soft_label, transform_params, change_mask = None):
        for i in range(len(idx)):
            if(idx[i] in self.record_softlb):
                soft_plabel = np.load(self.soft_labels[idx[i]])
            else:
                soft_plabel = np.load(self.soft_labels_init[idx[i]])
                self.record_softlb[idx[i]] = 1
            temp = self.img_size
            soft_plabel = torch.from_numpy(soft_plabel)
            soft_plabel = torch.nn.functional.interpolate(soft_plabel.unsqueeze(0), size=[temp[1],temp[0]], mode='bilinear', align_corners=True)[0]
            crop_i,crop_j,isFliped = transform_params
            if isFliped[i]:
                new_soft_label[i] = torch.flip(new_soft_label[i], dims=[2])

            if change_mask is not None:
                soft_before_labels = torch.argmax(soft_plabel, dim=0).numpy()
                img1 = ACDC.decode_target(soft_before_labels.copy()).astype(np.uint8)
                Image.fromarray(img1).save('test1.png')
                t = soft_plabel[:, crop_i[i]:crop_i[i]+768,crop_j[i]:crop_j[i]+768][:, change_mask[i]]
            soft_plabel[:, crop_i[i]:crop_i[i]+768,crop_j[i]:crop_j[i]+768] = new_soft_label[i]
            if change_mask is not None:
                t2 = soft_plabel[:, crop_i[i]:crop_i[i]+768,crop_j[i]:crop_j[i]+768][:, change_mask[i]]
                kk3 = new_soft_label[i][:, change_mask[i]]
                new_soft_label_d = torch.argmax(new_soft_label[i], dim=0).numpy()
                new_soft_label_d = ACDC.decode_target(new_soft_label_d).astype(np.uint8)
                Image.fromarray(new_soft_label_d).save('test2.png')
                soft_after_labels = soft_plabel[:, crop_i[i]:crop_i[i]+768,crop_j[i]:crop_j[i]+768]
                soft_after_labels_d = torch.argmax(soft_after_labels, dim=0).numpy()
                img1 = ACDC.decode_target(soft_after_labels_d).astype(np.uint8)
                Image.fromarray(img1).save('test3.png')

            soft_plabel = torch.nn.functional.interpolate(soft_plabel.unsqueeze(0), size=[temp[1] // 4,temp[0] // 4], mode='bilinear', align_corners=True)[0]
            if change_mask is not None:
                new_soft_label_d = torch.argmax(soft_plabel, dim=0).numpy()
                new_soft_label_d = ACDC.decode_target(new_soft_label_d).astype(np.uint8)
                Image.fromarray(new_soft_label_d).save('test4.png')
            # sdf = np.sum(soft_plabel != t)
            np.save(self.soft_labels[idx[i]], soft_plabel)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        return_dict = {}
        image = Image.open(self.images[index]).convert('RGB')
        # debug
        gt = None
        conf_plabel = None
        soft_plabel = None
        temp = image.size
        self.img_size = temp
        if(len(self.gts) > 0):
            gt = Image.open(self.gts[index])
            conf_plabel = Image.open(self.conf_plabels[index])

        if(len(self.soft_labels) > 0):
            if(index in self.record_softlb):
                soft_plabel = np.load(self.soft_labels[index])
            else:
                soft_plabel = np.load(self.soft_labels_init[index])
            soft_plabel = torch.from_numpy(soft_plabel)
            # t = torch.nn.functional.softmax(soft_plabel,dim=0)
            soft_plabel = torch.nn.functional.interpolate(soft_plabel.unsqueeze(0), size=[temp[1],temp[0]], mode='bilinear', align_corners=True)[0]
        if self.need_clean:
            train_type = "train"
            if self.images[index].find('val') > -1:
                train_type = "val"
            elif self.images[index].find('test') > -1:
                train_type = "test"
            t = self.images[index].replace(f'fog/{train_type}',f'fog/{train_type}_ref')
            t = t.replace('_anon.','_ref_anon.')
            clean_image = Image.open(t).convert('RGB')
        if self.test_only:
            target = Image.fromarray(np.zeros((image.size[1],image.size[0])))
        else:
            target = Image.open(self.targets[index])

        crop_i,crop_j = 0,0
        isFliped = False
        # target_bk = np.array(target)
        if self.transform:
            image, target = self.transform(image, target)
            # if target_bk[0] != target.numpy()[0]:
            #     isFliped = True
        target = self.encode_target(target)

        if type(self.transform.transforms[0]) == et.ExtRandomCrop:
            crop_i,crop_j = self.transform.transforms[0].i,self.transform.transforms[0].j
        if len(self.transform.transforms) > 2 and type(self.transform.transforms[2]) == et.ExtRandomHorizontalFlip:
            isFliped = self.transform.transforms[2].isFliped
        
        return_dict['labels'] = target
        return_dict['images'] = image
        return_dict['img_paths'] = self.images[index]
        return_dict['index'] = index
        if gt is not None:
            gt = np.array(gt)
            if type(self.transform.transforms[0]) == et.ExtRandomCrop:
                gt = gt[crop_i:crop_i+768,crop_j:crop_j+768]
                if isFliped:
                    gt = np.fliplr(gt)
            else:
                a = 1
            gt = self.encode_target(gt)
            return_dict['gts'] = gt
        if soft_plabel is not None:
            if type(self.transform.transforms[0]) == et.ExtRandomCrop:
                soft_plabel = soft_plabel[:, crop_i:crop_i+768,crop_j:crop_j+768]
                if isFliped:
                    soft_plabel = torch.flip(soft_plabel, dims=[2])
            return_dict['soft_plabels'] = soft_plabel
        if conf_plabel is not None:
            conf_plabel = np.array(conf_plabel)
            if type(self.transform.transforms[0]) == et.ExtRandomCrop:
                conf_plabel = conf_plabel[crop_i:crop_i+768,crop_j:crop_j+768]
                if isFliped:
                    conf_plabel = np.fliplr(conf_plabel)
            conf_plabel = self.encode_target(conf_plabel)
            return_dict['conf_plabels'] = conf_plabel
        if self.return_transform_p:
            return_dict['transform_params'] = (crop_i,crop_j,isFliped)
        if self.need_clean:
            if self.clean_plabels_path and os.path.exists(self.clean_plabels_path[index]):
                t = Image.open(self.clean_plabels_path[index])
            else:
                t = Image.fromarray(np.zeros((temp[1],temp[0])))
            clean_image, t = self.transform(clean_image, t)
            t = self.encode_target(t)
            return_dict['clean_images'] = clean_image
            return_dict['clean_plabels'] = t
        
        return return_dict

    def __len__(self):
        return len(self.images)
        # return min(50, len(self.images))

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

if __name__ == '__main__':
    ACDC('/media/user/storeDisk2/data/cwy/ACDC_Dataset',train_type='train')