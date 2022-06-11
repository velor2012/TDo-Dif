import json
import os
from collections import namedtuple
from PIL import Image
import torch
import torch.utils.data as data
import numpy as np
from natsort import natsorted
from cv2 import cv2
from utils import ext_transforms as et
import random
import numbers
import torchvision.transforms.functional as F
import glob
import re
class FoggyZurich(data.Dataset):
    """FoggyZurich <http://www.FoggyZurich-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/FoggyZurichScripts
    FoggyZurichClass = namedtuple('FoggyZurichClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        FoggyZurichClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        FoggyZurichClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        FoggyZurichClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        FoggyZurichClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        FoggyZurichClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        FoggyZurichClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        FoggyZurichClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        FoggyZurichClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        FoggyZurichClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        FoggyZurichClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        FoggyZurichClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        FoggyZurichClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        FoggyZurichClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        FoggyZurichClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        FoggyZurichClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        FoggyZurichClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        FoggyZurichClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        FoggyZurichClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        FoggyZurichClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        FoggyZurichClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        FoggyZurichClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        FoggyZurichClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        FoggyZurichClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        FoggyZurichClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        FoggyZurichClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        FoggyZurichClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        FoggyZurichClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        FoggyZurichClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        FoggyZurichClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        FoggyZurichClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        FoggyZurichClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        FoggyZurichClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        FoggyZurichClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        FoggyZurichClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        FoggyZurichClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_gray = [np.mean(c.color) for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, transform=None,dataset="light_unlabel",pseudo_label_dir=None,use_constract=False,return_transform_p = False,
    flow_dir='/media/user/storeDisk2/data/cwy/DenseMatching/test_map0.1',crop_bottom=True,part=None):
        self.return_transform_p = return_transform_p
        self.root = os.path.expanduser(root)
        self.images_dir = os.path.join(self.root, 'RGB')
        self.targets_dir = os.path.join(self.root, 'gt_labelIds')
        self.transform = transform
        self.use_constract = use_constract
        self.flow_dir = flow_dir
        self.crop_bottom = crop_bottom
        self.part =None
        self.total_part = None
        self.conf_plabels = []
        self.soft_labels = []
        if part is not None:
            assert len(part) == 2
            self.part = part[0]
            self.total_part = part[1]
        self.constract_transform = et.ExtCompose([
            # et.ExtResize((720,1280)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        self.images_map = None
        self.images = []
        self.targets = []
        self.dataset = dataset
        self.pseudo_label_dir = pseudo_label_dir
        if(self.dataset == "light_unlabel"):
            f = open(os.path.join(self.root,"lists_file_names/RGB_light_filenames.txt"),"r")   #璁剧疆鏂囦欢瀵硅薄
            line = f.readline()
            line = line[:-1]
            while line:             #鐩村埌璇诲彇瀹屾枃锟?
                if(line!=''):
                    self.images.append(os.path.join(self.root,line))
                    target_path = line.replace('RGB/','')
                    self.targets.append(os.path.join(self.targets_dir,target_path))
                line = f.readline()  #璇诲彇涓€琛屾枃浠讹紝鍖呮嫭鎹㈣锟?
                line = line[:-1]     #鍘绘帀鎹㈣绗︼紝涔熷彲浠ヤ笉锟?
            print('self.targets_dir %s'%(self.targets_dir))
            f.close() #鍏抽棴鏂囦欢
        elif(self.dataset == "fake_light_label"):
            if self.pseudo_label_dir is None or not os.path.exists(self.pseudo_label_dir):
                raise ValueError('pseudo label dir doesn\'t exist!')

            f = open(os.path.join(self.root,"lists_file_names/RGB_light_filenames.txt"),"r")   #璁剧疆鏂囦欢瀵硅薄
            line = f.readline()
            line = line[:-1]
            count = 0
            while line:             #鐩村埌璇诲彇瀹屾枃锟?
                if(line!=''):
                    self.images.append(os.path.join(self.root,line))
                    target_name = line.split('/')[-1]
                    self.targets.append(os.path.join(self.pseudo_label_dir,target_name))
                line = f.readline()  #璇诲彇涓€琛屾枃浠讹紝鍖呮嫭鎹㈣锟?
                line = line[:-1]     #鍘绘帀鎹㈣绗︼紝涔熷彲浠ヤ笉锟?
                count+=1
            print('self.targets_dir %s'%(self.targets_dir))
            f.close() #鍏抽棴鏂囦欢
        if(self.dataset == "medium_unlabel"):
            f = open(os.path.join(self.root,"lists_file_names/RGB_medium_filenames.txt"),"r")   #璁剧疆鏂囦欢瀵硅薄
            line = f.readline()
            line = line[:-1]
            while line:             #鐩村埌璇诲彇瀹屾枃锟?
                if(line!=''):
                    self.images.append(os.path.join(self.root,line))
                    target_path = line.replace('RGB/','')
                    self.targets.append(os.path.join(self.targets_dir,target_path))
                line = f.readline()  #璇诲彇涓€琛屾枃浠讹紝鍖呮嫭鎹㈣锟?
                line = line[:-1]     #鍘绘帀鎹㈣绗︼紝涔熷彲浠ヤ笉锟?
            print('self.targets_dir %s'%(self.targets_dir))
            f.close() #鍏抽棴鏂囦欢
        elif(self.dataset == "fake_medium_label"):
            if self.pseudo_label_dir is None or not os.path.exists(self.pseudo_label_dir):
                raise ValueError('pseudo label dir doesn\'t exist!')

            f = open(os.path.join(self.root,"lists_file_names/RGB_medium_filenames.txt"),"r")   #璁剧疆鏂囦欢瀵硅薄
            line = f.readline()
            line = line[:-1]
            count = 0
            while line:             #鐩村埌璇诲彇瀹屾枃锟?
                if(line!=''):
                    self.images.append(os.path.join(self.root,line))
                    target_name = line.split('/')[-1]
                    self.targets.append(os.path.join(self.pseudo_label_dir,target_name))
                line = f.readline()  #璇诲彇涓€琛屾枃浠讹紝鍖呮嫭鎹㈣锟?
                line = line[:-1]     #鍘绘帀鎹㈣绗︼紝涔熷彲浠ヤ笉锟?
                count+=1
            print('self.targets_dir %s'%(self.targets_dir))
            f.close() #鍏抽棴鏂囦欢
        elif(self.dataset == "test"):
            f = open(os.path.join(self.root,"lists_file_names/RGB_testv2_filenames.txt"),"r")   #璁剧疆鏂囦欢瀵硅薄
            line = f.readline()
            line = line[:-1]
            while line:             #鐩村埌璇诲彇瀹屾枃锟?
                if(line!=''):
                    self.images.append(os.path.join(self.root,line))
                    target_path = line.replace('RGB/','')
                    self.targets.append(os.path.join(self.targets_dir,target_path))
                line = f.readline()  #璇诲彇涓€琛屾枃浠讹紝鍖呮嫭鎹㈣锟?
                line = line[:-1]     #鍘绘帀鎹㈣绗︼紝涔熷彲浠ヤ笉锟?
            print('self.targets_dir %s'%(self.targets_dir))
            f.close() #鍏抽棴鏂囦欢
        elif(self.dataset == "all_unlabel"):
            for city in os.listdir(self.images_dir):
                img_dir = os.path.join(self.images_dir, city)
                target_dir = os.path.join(self.targets_dir, city)
                #瀵规枃浠跺悕鑷劧鎺掑簭锛岀劧鍚庡€掕繃鏉?
                b = natsorted(os.listdir(img_dir))
                b = b[::-1]
                for file_name in b:
                    self.images.append(os.path.join(img_dir, file_name))
        elif(self.dataset[:5] == "part_"):
            matchObj = re.match( r'part_(.*)', self.dataset, re.M|re.I)
            scene_id = int(matchObj.group(1))
            scenes = os.listdir(self.images_dir)
            city = scenes[scene_id-1]
            img_dir = os.path.join(self.images_dir, city)
            #瀵规枃浠跺悕鑷劧鎺掑簭锛岀劧鍚庡€掕繃鏉?
            b = natsorted(os.listdir(img_dir))
            b = b[::-1]
            for file_name in b:
                self.images.append(os.path.join(img_dir, file_name))
                    
        elif(self.dataset.find("all_fake_labels_part_")>=0):
            if self.pseudo_label_dir is None or not os.path.exists(self.pseudo_label_dir):
                raise ValueError('pseudo label dir doesn\'t exist!')
            matchObj = re.match( r'all_fake_labels_part_(.*)', self.dataset, re.M|re.I)
            scene_id = int(matchObj.group(1))
            scenes = os.listdir(self.images_dir)
            city = scenes[scene_id-1]
            img_dir = os.path.join(self.images_dir, city)

            b = natsorted(os.listdir(img_dir))
            b = b[::-1]
            for file_name in b:
                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(os.path.join(self.pseudo_label_dir,file_name))
        elif(self.dataset == "all_fake_labels"):
            if self.pseudo_label_dir is None or not os.path.exists(self.pseudo_label_dir):
                raise ValueError('pseudo label dir doesn\'t exist!')
            for city in os.listdir(self.images_dir):
                img_dir = os.path.join(self.images_dir, city)
                target_dir = os.path.join(self.targets_dir, city)
                #瀵规枃浠跺悕鑷劧鎺掑簭锛岀劧鍚庡€掕繃鏉?
                b = natsorted(os.listdir(img_dir))
                b = b[::-1]
                for file_name in b:
                    self.images.append(os.path.join(img_dir, file_name))
                    self.targets.append(os.path.join(pseudo_label_dir, file_name))
        elif(self.dataset == "lp_test"):
            if self.pseudo_label_dir is None or not os.path.exists(self.pseudo_label_dir):
                raise ValueError('pseudo label dir doesn\'t exist!')
            f = open(os.path.join(self.root,"lists_file_names/RGB_testv2_filenames.txt"),"r")   #璁剧疆鏂囦欢瀵硅薄
            line = f.readline()
            line = line[:-1]
            while line:             #鐩村埌璇诲彇瀹屾枃锟?
                if(line!=''):
                    self.images.append(os.path.join(self.root,line))
                    target_path = line.replace('RGB/','')
                    num = line[-7:-4]
                    num2 = '%03d' % (int(num)+1)
                    s = line[:-7] + num2 + line[-4:]
                    self.images.append(os.path.join(self.root,s))
                    # self.targets.append(os.path.join(self.targets_dir,target_path))
                line = f.readline()  #璇诲彇涓€琛屾枃浠讹紝鍖呮嫭鎹㈣锟?
                line = line[:-1]     #鍘绘帀鎹㈣绗︼紝涔熷彲浠ヤ笉锟?
            b = natsorted(self.images)
            self.images = b[::-1]
            for pa in self.images:
                mbasename = os.path.basename(pa)
                self.targets.append(os.path.join(self.pseudo_label_dir,mbasename))

            print('self.targets_dir %s'%(self.targets_dir))
            f.close() #鍏抽棴鏂囦欢

        #用于多线程生成伪标签
        if self.part is not None:
            slice_size = int(len(self.images) / self.total_part)
            #part从0开始
            _range = (self.part*slice_size,(self.part+1)*slice_size) if self.part != self.total_part - 1 else\
             ( self.part *slice_size,len(self.images))
            self.images = self.images[_range[0]:_range[1]]
            self.targets = self.targets[_range[0]:_range[1]]

        if self.pseudo_label_dir is not None:
            t = self.pseudo_label_dir[:self.pseudo_label_dir.rfind('/')]
            for img in self.images:
                file_name = os.path.basename(img)
                self.conf_plabels.append(os.path.join(t, 'pseudo_label_weighted', file_name))

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

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
        img_size = image.size

        conf_plabel = None
        soft_plabel = None
        temp = image.size

        if(self.dataset == "light_unlabel" or self.dataset == "all_unlabel" 
        or self.dataset == 'medium_unlabel' or self.dataset[:5] == "part_" or self.dataset == 'my_test'):
            target = Image.fromarray(np.zeros((image.size[1],image.size[0])))
        else:
            target = cv2.imread(self.targets[index],0)
            if self.crop_bottom and self.dataset != 'test':
                target[870:,:] = 0
            target = Image.fromarray(target)
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)

        crop_i,crop_j = 0,0
        isFliped = False
        if type(self.transform.transforms[0]) == et.ExtRandomCrop:
            crop_i,crop_j = self.transform.transforms[0].i,self.transform.transforms[0].j
        if len(self.transform.transforms) > 2 and type(self.transform.transforms[2]) == et.ExtRandomHorizontalFlip:
            isFliped = self.transform.transforms[2].isFliped

        if self.use_constract:
            if self.images_map is None:
                self.generate_img_hashmap('datasets/targetImgs') #Foggy Zurich
            basename = os.path.basename(self.images[index]).split('.png')[0]
            
            while basename not in self.images_map:
                index -= 1
                basename = os.path.basename(self.images[index]).split('.png')[0]
            
            # image2 = Image.open(self.images[index]).convert('RGB')
            closer_image = Image.open(self.images_map[basename]).convert('RGB')
            fake_t = Image.fromarray(np.zeros((img_size[1],img_size[0])))
            closer_image_basename = os.path.basename(self.images_map[basename]).split('.png')[0]
            xymap = np.load(os.path.join(self.flow_dir,'{}_xymap_back.npy'.format(closer_image_basename))).astype(np.int32)
            closer_image, _ = self.constract_transform(closer_image, fake_t)
            w = target.shape[-1]
            return_dict['closer_img'] = closer_image
            return_dict['xymap'] = xymap[crop_i:crop_i+w,crop_j:crop_j+w]
            
        return_dict['labels'] = target
        return_dict['images'] = image
        return_dict['img_paths'] = self.images[index]
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
        return return_dict
    def __len__(self):
        return len(self.images)
        # return min(100, len(self.images))

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def rangdom_crop(self,image_tensor,xymap,ourtput_size):
        size = None
        if isinstance(ourtput_size, numbers.Number):
            size = (int(ourtput_size), int(ourtput_size))
        else:
            size = ourtput_size
        _,h, w = image_tensor.size()
        th, tw = size
        if w == tw and h == th:
            return 0, 0, h, w

        # i = random.randint(0, h - th)
        # j = random.randint(0, w - tw)
        i = int((h-th)/2)
        j = int((w-tw)/2)
        return image_tensor[:,i:i+th,j:j+tw], xymap[i:i+th,j:j+tw,:]
    
    # far_img_path->closer_img_path
    def generate_img_hashmap(self,image_path):
        self.images_map = {}
        imagelist = glob.glob(os.path.join(image_path,'*.png'))
        imagelist_sorted = natsorted(imagelist)
        for index in range(len(imagelist_sorted)):
            if index ==0:
                continue
            far_img_path = imagelist_sorted[index-1]
            closer_img_path = imagelist_sorted[index]
            temp_string = os.path.basename(far_img_path)
            temp_string = temp_string.split('.png')[0]
            far_img_id = temp_string[-6:]
            temp_string2 = os.path.basename(closer_img_path)
            temp_string2 = temp_string2.split('.png')[0]
            closer_img_id = temp_string2[-6:]
            a = int(far_img_id)
            b = int(closer_img_id)
            c = os.path.basename(far_img_path)[:10]
            d = os.path.basename(closer_img_path)[:10] 
            if(a == b - 1 and c == d):
                self.images_map[temp_string] = closer_img_path