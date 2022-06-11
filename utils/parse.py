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
Date: 2021-04-29 21:02:43
Description: file content
'''
import argparse
def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--source_data_root", type=str, default='~/SyntheticCityScapes',
                        help="path to Dataset")
    parser.add_argument("--source_dataset", type=str, default='SyntheticCityscapes',
                        choices=['voc', 'cityscapes','FoggyDriving','FoggyZurich','SyntheticCityscapes', 'ACDC'], help='Name of dataset')
    parser.add_argument("--target_data_root", type=str, default='~/Foggy_Zurich',
                        help="path to Dataset")
    parser.add_argument("--target_dataset", type=str, default='FoggyZurich',
                        choices=['voc', 'cityscapes','FoggyDriving','FoggyZurich','SyntheticCityscapes','SyntheticCityscapesAll', 'ACDC'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='refineNet',
                        choices=['rf101_contrastive','refineNet', 'hrnet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=8, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--usegpu", action='store_true', default=False)
    parser.add_argument("--train_type", type=str, default='CBST_base',
                        choices=['CBST_base',  'CRST_base',
                                 'CRST_sp', 'CRST_sp_lp', 'CRST_constract',
                                 'CRST_sp_lp_constract','CRST_sp_constract',
                                 "CRST_sp_with_loss","CRST_sp_with_loss_lp","CRST_sp_with_loss_lp_constract"
                                 ], help='train type')
    parser.add_argument("--round_idx", type=int, default=0,
                        help="cur training round (default: 0)")
    parser.add_argument("--train_dataset_type", type=str, default='light',
                        choices=['light',  'medium'], help='train dataset type, including \'light\' and \'meidum\'')
    parser.add_argument("--light_pseudo_label_path", type=str, default='',help='only useful when train dataset type set \'medium\'')
    parser.add_argument("--use_teacher", action='store_true', default=False)
    parser.add_argument("--weight", action='store_true', default=False)
    parser.add_argument("--temper_scaling", action='store_true', default=False)
    parser.add_argument("--refine_lb", action='store_true', default=False)
    parser.add_argument("--soft_label", action='store_true', default=False)
    parser.add_argument("--net_momentum",type=float, default=0.99)
    # parser.add_argument("--moco_queue",type=int, default=100, help='queue size for each class, default 100')
    parser.add_argument("--low_dim",type=int, default=128, help='feature dimension for constract learning, default 100')
    parser.add_argument("--sp_weight",type=float, default=10)
    parser.add_argument("--con_weight",type=float, default=0.1)
    parser.add_argument("--temp",type=float, default=5)
    parser.add_argument("--total_round", type=int, default=2)
    parser.add_argument("--save_result_path", type=str, default='results')
    parser.add_argument("--save_model_prefix", type=str, default='')
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--epoch_one_round", type=int, default=5,
                        help="epoch number in one round (default: 5)")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--kld_weight", type=float, default=0.1,
                        help="kld_weight (default: 0.1)")
    parser.add_argument("--unconf_weight", type=float, default=0.1,
                        help="unconf_weight (default: 0.1)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--contrast_refine", action='store_true', default=False,
                        help='calculate contrast loss with softlabel refine (default: False)')
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy','focal_loss', 'SCELoss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 1000)")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="epoch interval for eval (default: 1)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    # 分布式
    parser.add_argument('--local_rank', type=int, default=0,help='node rank for distributed training')
#TODO 加上参数选择是否用多视图的伪标签挑选
    # stage1
    parser.add_argument("--init_target_portion", type=float, default=0.2,
                        help="init pseudo label portion (default: 0.2)")
    # parser.add_argument("--max_target_portion", type=float, default=0.5,
    #                     help="max pseudo label portion (default: 0.5)")
    parser.add_argument("--target_portion_step", type=float, default=0.1,
                        help="pseudo label portion grows after every epoch (default: 0.05)")
    parser.add_argument("--ds_rate", type=int, default=4,
                        help="the rate for downsampling the pseudo label while caculate the class-wise confidence (default: 4)")

   #stage2
    parser.add_argument("--seg_num", type=int, default=500,
                        help="superpixels segmentation nums (default: 500)")

    parser.add_argument("--xymap_dir", type=str, default='/media/user/storeDisk2/data/cwy/DenseMatching/xymap_720p',
                        help="xymap dir used in stage3")
    #其他
    parser.add_argument("--skip_thresh_gen", action='store_true', default=False,
                    help="skip threshhold generation in stage1")
    parser.add_argument("--skip_p_gen", action='store_true', default=False,
                    help="skip pseudo labels generation in stage1")
    parser.add_argument("--skip_sp_extend", action='store_true', default=False,
                    help="skip superpixel extend for pseudo labels in stage2")
    parser.add_argument("--skip_lp_extend", action='store_true', default=False,
                    help="skip label propegating for pseudo labels in stage3")
    parser.add_argument("--only_generate", action='store_true', default=False,
                    help="only generate pseudo labels before training")
    return parser
