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
Date: 2021-04-29 21:10:14
Description: generate the dataset used in self-training
'''
from numpy.core.fromnumeric import resize
from torchvision import transforms
from datasets import FoggyZurich
from utils import ext_transforms as et

#DONE 修改该函数，通过stage改变train_transform和相关参数
def get_medium_zurich_self_training_dataset(img_dataset_root,stage_index,pseudo_label_dir=None,
crop_size=None,use_constract=False,part=None):
    train_transform = None
    if stage_index == 1:
        train_transform = et.ExtCompose([
                # et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    elif stage_index == 2:
        train_transform = et.ExtCompose([
                # et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    elif stage_index == 3:
        train_transform = et.ExtCompose([
                # et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        train_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(crop_size, crop_size)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    #stage 1
    if(stage_index == 1):
        if part is not None:
            train_dst = FoggyZurich(root=img_dataset_root, transform=train_transform,dataset="medium_unlabel",part=part)
        else:
            train_dst = FoggyZurich(root=img_dataset_root, transform=train_transform,dataset="medium_unlabel")
    elif stage_index == 2:
        if part is not None:
            train_dst = FoggyZurich(root=img_dataset_root, transform=train_transform,dataset="all_unlabel",part=part)
        else:
            train_dst = FoggyZurich(root=img_dataset_root, transform=train_transform,dataset="all_unlabel")
    elif stage_index == 3:
        train_dst = [
            FoggyZurich(root=img_dataset_root, transform=train_transform,dataset="all_fake_labels_part_1",pseudo_label_dir=pseudo_label_dir,crop_bottom=False),
            FoggyZurich(root=img_dataset_root, transform=train_transform,dataset="all_fake_labels_part_2",pseudo_label_dir=pseudo_label_dir,crop_bottom=False),
            FoggyZurich(root=img_dataset_root, transform=train_transform,dataset="all_fake_labels_part_3",pseudo_label_dir=pseudo_label_dir,crop_bottom=False),
            FoggyZurich(root=img_dataset_root, transform=train_transform,dataset="all_fake_labels_part_4",pseudo_label_dir=pseudo_label_dir,crop_bottom=False)
        ]
    elif stage_index == 4 and pseudo_label_dir is not None:
        train_dst = FoggyZurich(root=img_dataset_root, transform=train_transform,dataset="fake_medium_label",
        pseudo_label_dir=pseudo_label_dir,use_constract = use_constract, return_transform_p=True, flow_dir = xymap_dir)
    else:
        raise ValueError('parameters for function \'get_zurich_self_training_dataset\' are wrong!')
    return train_dst