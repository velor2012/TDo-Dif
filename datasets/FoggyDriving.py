import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import copy
from utils import ext_transforms as et

class FoggyDriving(data.Dataset):
    """FoggyDriving <http://www.FoggyDriving-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/FoggyDrivingScripts
    FoggyDrivingClass = namedtuple('FoggyDrivingClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        FoggyDrivingClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        FoggyDrivingClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        FoggyDrivingClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        FoggyDrivingClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        FoggyDrivingClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        FoggyDrivingClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        FoggyDrivingClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        FoggyDrivingClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        FoggyDrivingClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        FoggyDrivingClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        FoggyDrivingClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        FoggyDrivingClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        FoggyDrivingClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        FoggyDrivingClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        FoggyDrivingClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        FoggyDrivingClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        FoggyDrivingClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        FoggyDrivingClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        FoggyDrivingClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        FoggyDrivingClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        FoggyDrivingClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        FoggyDrivingClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        FoggyDrivingClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        FoggyDrivingClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        FoggyDrivingClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        FoggyDrivingClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        FoggyDrivingClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        FoggyDrivingClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        FoggyDrivingClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        FoggyDrivingClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        FoggyDrivingClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        FoggyDrivingClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        FoggyDrivingClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        FoggyDrivingClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        FoggyDrivingClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, mode='fine', target_type='semantic', transform=None,test_only=True,plabel_dir =None):
        self.root = os.path.expanduser(root)
        self.target_type = 'labelIds'
        self.images_dir = os.path.join(self.root, 'leftImg8bit')

        self.test_only = test_only
        self.plabel_dir = plabel_dir
        self.transform = transform
        self.transform_c = copy.deepcopy(transform)
        self.transform_c.transforms = self.transform_c.transforms[1:]
        self.images = []
        self.targets = []

        # add 2022_5_16
        self.img_size = (1920, 1080)
        self.conf_plabels = []
        self.soft_labels = []
        self.soft_labels_init = []
        self.record_softlb = {}

        f = open(os.path.join(self.root,"lists_file_names/leftImg8bit_testall_filenames.txt"),"r")   #设置文件对象
        line = f.readline()
        line = line[:-1]
        while line:             #直到读取完文�?
            if(line!=''):
                self.images.append(os.path.join(self.root,line))
                a = line.split('test_extra')
                if(len(line.split('test_extra'))>1):
                    self.mode = 'gtCoarse'
                else:
                    self.mode = 'gtFine'
                self.targets_dir = os.path.join(self.root, self.mode)
                if not self.test_only and self.plabel_dir is not None:
                    target_path = os.path.basename(line)
                    self.targets.append(os.path.join(self.plabel_dir,target_path))

                    t = self.plabel_dir[:self.plabel_dir.rfind('/')]
                    self.conf_plabels.append(os.path.join(t, 'pseudo_label_weighted', target_path))
                    self.soft_labels_init.append(os.path.join(t, 'soft_labels',  target_path.replace('png','npy')))
                    if not os.path.exists(os.path.join(t, 'soft_labels_init')):
                        os.mkdir(os.path.join(t, 'soft_labels_init'))
                    self.soft_labels.append(os.path.join(t, 'soft_labels_init',  target_path.replace('png','npy')))

                else:
                    target_path = '{}_{}_{}{}'.format(line.split('_leftImg8bit')[0].replace('leftImg8bit/',''),
                            self.mode,self.target_type,line.split('_leftImg8bit')[1])
                    self.targets.append(os.path.join(self.targets_dir,target_path))
            line = f.readline()  #读取一行文件，包括换行�?
            line = line[:-1]     #去掉换行符，也可以不�?
        f.close() #关闭文件

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
                img1 = FoggyDriving.decode_target(soft_before_labels.copy()).astype(np.uint8)
                Image.fromarray(img1).save('test1.png')
                t = soft_plabel[:, crop_i[i]:crop_i[i]+768,crop_j[i]:crop_j[i]+768][:, change_mask[i]]
            if type(self.transform.transforms[1]) == et.ExtRandomCrop:
                soft_plabel[:, crop_i[i]:crop_i[i]+768,crop_j[i]:crop_j[i]+768] = new_soft_label[i]
            else:
                soft_plabel = new_soft_label[i]

            if change_mask is not None:
                t2 = soft_plabel[:, crop_i[i]:crop_i[i]+768,crop_j[i]:crop_j[i]+768][:, change_mask[i]]
                kk3 = new_soft_label[i][:, change_mask[i]]
                new_soft_label_d = torch.argmax(new_soft_label[i], dim=0).numpy()
                new_soft_label_d = FoggyDriving.decode_target(new_soft_label_d).astype(np.uint8)
                Image.fromarray(new_soft_label_d).save('test2.png')
                soft_after_labels = soft_plabel[:, crop_i[i]:crop_i[i]+768,crop_j[i]:crop_j[i]+768]
                soft_after_labels_d = torch.argmax(soft_after_labels, dim=0).numpy()
                img1 = FoggyDriving.decode_target(soft_after_labels_d).astype(np.uint8)
                Image.fromarray(img1).save('test3.png')

            soft_plabel = torch.nn.functional.interpolate(soft_plabel.unsqueeze(0), size=[temp[1] // 4,temp[0] // 4], mode='bilinear', align_corners=True)[0]
            if change_mask is not None:
                new_soft_label_d = torch.argmax(soft_plabel, dim=0).numpy()
                new_soft_label_d = FoggyDriving.decode_target(new_soft_label_d).astype(np.uint8)
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
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        crop_i,crop_j = 0,0
        isFliped = False
        temp = image.size
        # resize_num = 1024
        # temp = (image.size[0] * resize_num // image.size[1], resize_num)
        # # if temp[0] == 819 and temp[1] == 1024:
        # #     sd = 123
        # if temp[0] < 1024 or temp[1] < 1024:
        #     temp = (resize_num, image.size[1] * resize_num // image.size[0])
        self.img_size = image.size
        return_dict = {}
        conf_plabel = None
        soft_plabel = None
        if(len(self.soft_labels) > 0):
            if(index in self.record_softlb):
                soft_plabel = np.load(self.soft_labels[index])
                # soft_plabel = torch.ones(19,temp[1],temp[0])
                # soft_plabel /= 19
                # np.save(self.soft_labels[index], soft_plabel.numpy())
            else:
                soft_plabel = np.load(self.soft_labels_init[index])
            soft_plabel = torch.from_numpy(soft_plabel)
            soft_plabel = torch.nn.functional.interpolate(soft_plabel.unsqueeze(0), size=[temp[1],temp[0]], mode='bilinear', align_corners=True)[0]
            conf_plabel = np.array(Image.open(self.conf_plabels[index]))
            conf_plabel = torch.from_numpy(conf_plabel)
            conf_plabel = torch.nn.functional.interpolate(conf_plabel.unsqueeze(0).unsqueeze(0), size=[temp[1],temp[0]], mode='nearest')[0][0]

        if self.transform:
            if not self.test_only and image.size[-1] < 768:
                image, target = self.transform(image, target)
            else:
                image, target = self.transform(image, target)
            if not self.test_only:
                if type(self.transform.transforms[1]) == et.ExtRandomCrop  and temp[-1] >= 768:
                    crop_i,crop_j = self.transform.transforms[1].i,self.transform.transforms[1].j
                if  temp[-1] >= 768 and len(self.transform.transforms) > 2 and type(self.transform.transforms[3]) == et.ExtRandomHorizontalFlip:
                    isFliped = self.transform.transforms[3].isFliped
                elif temp[-1] < 768 and self.transform.transforms[2] == et.ExtRandomHorizontalFlip:
                    isFliped = self.transform.transforms[2].isFliped
                return_dict['transform_params'] = (crop_i,crop_j,isFliped)
        
        if soft_plabel is not None:
            if type(self.transform.transforms[1]) == et.ExtRandomCrop:
                soft_plabel = soft_plabel[:, crop_i:crop_i+768,crop_j:crop_j+768]
                if isFliped:
                    soft_plabel = torch.flip(soft_plabel, dims=[2])
            return_dict['soft_plabels'] = soft_plabel
        if conf_plabel is not None:
            conf_plabel = np.array(conf_plabel)
            if type(self.transform.transforms[1]) == et.ExtRandomCrop:
                t = conf_plabel.copy()
                conf_plabel = conf_plabel[crop_i:crop_i+768,crop_j:crop_j+768]
                if conf_plabel.shape[-1] != conf_plabel.shape[-2]:
                    asd = 12
                if isFliped:
                    conf_plabel = np.fliplr(conf_plabel)
            conf_plabel = self.encode_target(conf_plabel)
            return_dict['conf_plabels'] = conf_plabel

        target = self.encode_target(target)
        return_dict['images'] = image
        return_dict['labels'] = target
        return_dict['img_paths'] = self.images[index]
        return_dict['index'] = index
        return return_dict

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data
