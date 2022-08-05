from itertools import cycle
from tqdm import tqdm
import network
import utils
import os
import random
import numpy as np
from torch.utils import data
from utils.MLD_CE import cal_kld_loss
from utils import SCELoss
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
from stage1 import STAGE1
from stage2 import STAGE2
from stage3 import STAGE3
from PIL import Image
from cv2 import cv2
from utils import *
# from utils.PixelContrastLoss_with_refine import PixelContrastLoss
import time

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
def val(model, device, loader,metrics, args):
    """
    @description: Do validation and return specified samples
    @param {segmantation model} model
    @param {*} device
    @param {*} loader
    @param {*} metrics
    @param {dict} args
    @return {*} conf_dict, pred_cls_num,score, ret_samples
    """
    metrics.reset()
    ret_samples = []
    result_dir = os.path.join(args.save_result_path,'test')
    if args.save_val_results:
        if not os.path.exists(args.save_result_path):
            os.mkdir('results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
    img_id = 0

    ## output of deeplab is logits, not probability
    softmax2d = nn.Softmax2d()
    interp = None
    with torch.no_grad():
        for i, datas in enumerate(tqdm(loader)):
            image_names = None
            images = datas['images']
            labels = datas['labels']
            if 'img_paths' in datas:
                image_names = datas['img_paths']
            if 'gts' in datas:
                gts = datas['gts']
                gts = gts.to(device, dtype=torch.long)
            if 'conf_plabels' in datas:
                conf_plabels = datas['conf_plabels']
            if 'transform_params' in datas:
                (off_i,off_j,isFliped) = datas['transform_params']
            images = images.to(device, dtype=torch.float32)
            if args.use_teacher:
                res = model(images,use_teacher=True)
                # outputs,_ = model(images, update_teacher = False,\ 
                #                 plabels = labels, update_prototype = False,\ 
                #                 refine_outputs = True, labels = None, conf_plabels = conf_plabels)
            else:
                res = model(images)
            outputs, feats = res[0], res[-1]

            labels_inter = torch.nn.functional.interpolate(labels.unsqueeze(1).float(),
                                                 (feats.shape[2], feats.shape[3]), mode='nearest').long()
            labels_inter = labels_inter.squeeze(1)

            ## upsampling layer
            # if interp is None:
            interp = torch.nn.Upsample(size=labels.shape[1:], mode='bilinear')
            outputs_interpolated = softmax2d(interp(outputs)).cpu().numpy()
            # outputs_interpolated = torch.nn.functional.interpolate(
            #     outputs, size=labels.shape[1:], mode="bilinear", align_corners=False
            # ).cpu().numpy()
            preds = np.argmax(outputs_interpolated,axis=1)
            prohs = outputs_interpolated.copy()
            targets = labels.cpu().numpy()
            # targets = loader.dataset.encode_target(targets)
            # targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            if args.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]
                    proh = prohs[i]
                    file_name = None
                    if(image_names is not None):
                        file_name = image_names[i].split('/')[-1]
                        file_name = file_name[:file_name.rfind('.')]
                    else:
                        file_name = img_id
                        

                    # entropy_map_i = (entropy_map[i]* 255).astype(np.uint8)
                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    # Image.fromarray(pred.astype(np.uint8)).save('%s/%s.png' % (result_dir,file_name))
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)
                    target = target.astype(np.uint8)
                    # pred = pred.astype(np.uint8)

                    Image.fromarray(image).save('%s/%s_image.png' % (result_dir,file_name))
                    Image.fromarray(target).save('%s/%s_target.png' % (result_dir,file_name))
                    Image.fromarray(pred).save('%s/%s_pred.png' % (result_dir,file_name))
                    
                    # Image.fromarray(entropy_map_i).save('%s/%s_entropy.png' % (result_dir,file_name))
                    # np.save('results/%d_proh'%img_id,proh[i])
                    # proh = np.max(proh,axis=0)
                    # proh = (proh * 255).astype(np.uint8)
                    # Image.fromarray(proh).save('%s/%s_proh.png' % (result_dir,file_name))
                    # for class_id in range(len(proh)):
                    #     Image.fromarray(proh[class_id]).save('%s/%s_pred_class_%s.png' % (result_dir,file_name,class_id))
                    img_id += 1

        score = metrics.get_results()

    return score, ret_samples
def main():
    opts = get_argparser().parse_args()
    opts.num_classes = 19
    total_epoch = opts.epoch_one_round * opts.total_round

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    if opts.usegpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    # dt.misc.set_seed(4)
    if(str(device)=='cuda'):
        torch.backends.cudnn.deterministic = True
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    
    source_train_dst, source_val_dst = get_dataset(opts.source_dataset,opts.source_data_root,opts.crop_size,opts.test_only)
    _, target_test_dst = get_dataset(opts.target_dataset,opts.target_data_root,opts.crop_size,opts.test_only)
    if opts.test_only:
        source_train_loader = []
    else:
        source_train_loader = data.DataLoader(source_train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0)
        # use the temporally information , can't shuffle. In first round, 
        # all pixels in target dataset's labels are 1, means the labels are fake 
    source_val_loader = data.DataLoader(
        source_val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=0)
    target_test_loader = data.DataLoader(
        target_test_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=0)
    #NOTE DEBUG
    tgt_light_dst = None
    tgt_light_loader = None
    target_dst_stage1 = None
    if opts.target_dataset == 'FoggyZurich':
        if opts.train_dataset_type == 'medium':
            if not os.path.exists(opts.light_pseudo_label_path):
                raise('light_pseudo_label_path doesn\'t exist')
            tgt_light_dst = get_zurich_self_training_dataset(opts.target_data_root,stage_index=4,pseudo_label_dir=opts.light_pseudo_label_path
                            ,crop_size=opts.crop_size)
            tgt_light_loader = data.DataLoader(tgt_light_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0)
            target_dst_stage1 = get_medium_zurich_self_training_dataset(opts.target_data_root,stage_index=1)
        else:
            target_dst_stage1 = get_zurich_self_training_dataset(opts.target_data_root,stage_index=1)
    else:
        target_dst_stage1, _ = get_dataset(opts.target_dataset,opts.target_data_root,opts.crop_size,opts.test_only)

    target_train_loader = data.DataLoader(
        target_dst_stage1, batch_size=opts.batch_size, shuffle=False, num_workers=0)
    if opts.target_dataset == 'FoggyZurich' and opts.train_dataset_type == 'medium':
        print("Source Dataset: %s, Source Train set: %d, Source Val set: %d, Target Train set:%d, Target light set:%d,  Target Test set: %d " %
            (opts.source_dataset, len(source_train_dst), len(source_val_dst),len(target_dst_stage1), len(tgt_light_dst), len(target_test_dst)))
    else:
        print("Source Dataset: %s, Source Train set: %d, Source Val set: %d, Target Train set:%d, Target Test set: %d " %
            (opts.source_dataset, len(source_train_dst), len(source_val_dst),len(target_dst_stage1), len(target_test_dst)))

    # Set up model
    model_map = {
        'refineNet':network.getRefineNet,
        "rf101_contrastive":network.rf101_contrastive,
        "hrnet":network.getHRNet
    }

    # if opts.use_teacher:
    model_base = model_map[opts.model](num_classes=opts.num_classes)
    model_teacher = model_map[opts.model](num_classes=opts.num_classes)
    model_base.apply(inplace_relu)
    model_teacher.apply(inplace_relu)
    model = network.MomentumNet(model_base, model_teacher, opts)

    metrics = StreamSegMetrics(opts.num_classes)

    from network.refineNet import get_encoder_and_decoder_params
    enc_params, dec_params = None,None
    # if opts.use_teacher:
    #     enc_params, dec_params = get_encoder_and_decoder_params(model_base)
    # else:
    enc_params, dec_params = get_encoder_and_decoder_params(model_base)

    # Optimisers
    optimizers = [
        torch.optim.SGD(enc_params,lr=0.1*opts.lr,weight_decay=opts.weight_decay,momentum=0.9),
        torch.optim.SGD(dec_params,lr=opts.lr,weight_decay=opts.weight_decay,momentum=0.9)
    ]
    schedulers = [
        # PolyLR(optimizers[0], max_iter, power=0.9),
        # PolyLR(optimizers[1], max_iter, power=0.9)
        torch.optim.lr_scheduler.LambdaLR(optimizers[0],lr_lambda = lambda epoch:(1.0 - 1.0 * epoch / opts.epoch_one_round) ** 0.9),
        torch.optim.lr_scheduler.LambdaLR(optimizers[1],lr_lambda = lambda epoch:(1.0 - 1.0 * epoch / opts.epoch_one_round) ** 0.9),
    ]

    if opts.loss_type == 'SCELoss':
        criterion = SCELoss(alpha=0.1,beta=1,ignore_index=255)
    elif opts.loss_type == 'focal_loss':
        criterion = FocalLoss(ignore_index=255, size_average=True)
        # criterion = Criterion_cons(gamma=2, ignore_index=255)
        # criterion = FocalLossV1(ignore_index=255, mode = 'multiclass')
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255)

    criterion_constract = ConstractLoss()
    spatial_loss = SpatialLoss()
    cur_epochs = None
    cur_itrs = None
    def save_ckpt(path,use_teacher=False):
        """ save current model
        """
        # my_model = model.module
        if use_teacher:
            my_model = model_teacher
        else:
            my_model = model_base
        torch.save({
            #add by cwy
            "cur_epochs":cur_epochs,
            #end add
            "cur_itrs": cur_itrs,
            # "model_state": my_model.module.state_dict(),
            "model_state": my_model.state_dict(),
            "optimizer_enc_state": optimizers[0].state_dict(),
            "optimizer_dec_state": optimizers[1].state_dict(),
            "scheduler_enc_state": schedulers[0].state_dict(),
            "scheduler_dec_state": schedulers[1].state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    print(f"opts.ckpt:{opts.ckpt}")
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        if opts.use_teacher:
            load_model_from_dict(model_base, checkpoint["model_state"])
            load_model_from_dict(model_teacher, checkpoint["model_state"].copy())
        else:
            load_model_from_dict(model_base, checkpoint["model_state"])
        if(str(device)=='cuda'):
            model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizers[0].load_state_dict(checkpoint["optimizer_enc_state"])
            optimizers[1].load_state_dict(checkpoint["optimizer_dec_state"])
            schedulers[0].load_state_dict(checkpoint["scheduler_enc_state"])
            schedulers[1].load_state_dict(checkpoint["scheduler_dec_state"])
            #need to add this,otherwise if you set a small total_itrs
            # ,while retraining set a large itrs,it will get error
            # add by cwy
            cur_epochs = checkpoint["cur_epochs"]
            # end add
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        if(str(device)=='cuda'):
            model = nn.DataParallel(model)
        model.to(device)

    if opts.test_only:
        model.eval()
        test_score, _ = val(model,device,target_test_loader,metrics,opts)
        print(metrics.to_str(test_score))
        print(metrics.get_results())

        return

    interval_loss2 = 0
    sp_loss_sum = 0
    constract_loss_sum = 0
    target_all_loader = None
    save_round_eval_path = None
    use_sp_et = False
    use_lp_et = False 
    use_constract = False
    use_kld = False
    use_spatial_loss = False
    final_path = None
    begin = 0
    # if opts.train_dataset_type == 'light':
    #     begin = 1
    round_idx = opts.round_idx
    model.eval()
    if opts.train_type == 'CRST_base':
        use_kld = True
    elif opts.train_type == 'CRST_sp':
        use_kld = True
        use_sp_et = True
    elif opts.train_type == 'CRST_sp_lp':
        use_kld = True
        use_sp_et = True
        # use_constract = True
        # use_spatial_loss = True
        use_lp_et = True
    elif opts.train_type == 'CRST_constract':
        use_kld = True
        use_constract = True
    elif opts.train_type == 'CRST_sp_lp_constract':
        use_kld = True
        use_sp_et = True
        use_lp_et = True
        use_constract = True
    elif opts.train_type == 'CRST_sp_constract':
        use_kld = True
        use_sp_et = True
        use_constract = True
    elif opts.train_type == 'CRST_sp_with_loss':
        use_kld = True
        use_sp_et = True
        use_spatial_loss = True
    elif opts.train_type == 'CRST_sp_with_loss_lp':
        use_kld = True
        use_sp_et = True
        use_spatial_loss = True
        use_lp_et = True
    elif opts.train_type == 'CRST_sp_with_loss_lp_constract':
        use_kld = True
        use_sp_et = True
        use_constract = True
        use_spatial_loss = True
        use_lp_et = True

    save_round_eval_path = os.path.join(opts.save_result_path,'{}_{}_round_{}_{}_{}'.format(opts.target_dataset.lower(), opts.save_model_prefix, round_idx,opts.train_dataset_type,opts.train_type))
    # save_round_eval_path = 'results/foggyzurich_without_soft_with_refine_round_678_light_CRST_sp_lp'
    stage1 = STAGE1(round_idx,opts,save_round_eval_path)
    # # NOTE for only debug ↓ use_resume
    if (opts.skip_thresh_gen):
        print('load thresh')
        cls_thresh = np.load(stage1.save_stats_path + '/cls_thresh_round' + str(round_idx) + '.npy')
    else:
        conf_dict, pred_cls_num = stage1.cal_class_wise_confidence_infor(model,device,target_train_loader,opts)
        # return
        cls_thresh = stage1.cal_threshold_kc(conf_dict=conf_dict,pred_cls_num=pred_cls_num,args=opts)
        # del target_train_loader
    if (not opts.skip_p_gen):
        loaders = []
        if opts.target_dataset == 'FoggyZurich' and use_lp_et:
            thread_nums = 4
            for i in range(thread_nums):
                dataset = get_zurich_self_training_dataset(opts.target_data_root,stage_index=2,part=[i,thread_nums])
                loader = data.DataLoader(
                    dataset, batch_size=1, shuffle=False, num_workers=0)
                loaders.append(loader)
        else:
            loaders.append(target_train_loader)
        print('######## stage1.label_selection ##########################')
        stage1.label_selection(model,loaders,cls_thresh,device,opts)


    # # 超像素扩充
    if opts.target_dataset == 'FoggyZurich':
        base_save_sp_path ='FoggyZurich_superpixel_save_path'
    else:
        base_save_sp_path = 'Driving_superpixel_save_path'
    if not os.path.exists(base_save_sp_path):
        os.makedirs(base_save_sp_path)
    save_sp_path = os.path.join(base_save_sp_path,'{}_superpixel'.format(opts.seg_num))
    stage2 = STAGE2(round_idx, opts,save_round_eval_path, save_sp_path)
    if use_sp_et and not opts.skip_sp_extend:
        path = stage1.save_origin_wpred_path
        plabel_path = stage1.save_pseudo_label_weighted_path

        #需要把目标域所有图片放到stage1.save_img_path中
        stage2.extend_pseudo_by_superpixels_m(opts,plabel_path,path,stage1.save_img_path)

    #stage3
    stage3 = STAGE3(round_idx,opts,save_round_eval_path)
    if opts.target_dataset == 'FoggyZurich' and not opts.skip_lp_extend and use_lp_et:
    # if use_lp_et and opts.target_dataset == 'FoggyZurich' and not opts.skip_lp_extend:
        print('######## stage3.label_selection ##########################')
        flow_dir = opts.xymap_dir
        datasets = get_zurich_self_training_dataset(opts.target_data_root,stage_index=3,pseudo_label_dir=stage2.save_multiview_labels_intra_path,
            crop_size=opts.crop_size, xymap_dir = opts.xymap_dir)
        loaders = []
        for dataset in datasets:
            print(len(dataset))
            loaders.append(data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0))  
        stage3.label_propagation(model,device,loaders,flow_dir,opts)    

    if opts.only_generate:
        print("done all generation!")
        return

    if opts.train_type.find('lp') >= 0:
        final_path = stage3.save_concat_label_path
    elif opts.train_type.find('sp') >= 0:
        final_path = stage2.save_multiview_labels_intra_path
    else:
        final_path = stage1.save_pseudo_label_weighted_path

    print('psuedo label path:{}'.format(final_path))
    print('use_kld:{}'.format(str(use_kld)))
    print('use_sp_et:{}'.format(str(use_sp_et)))
    print('use_spatial_loss:{}'.format(str(use_spatial_loss)))
    print('sp_weight:{}'.format(str(opts.sp_weight)))

    if opts.target_dataset == 'FoggyZurich':
        func = get_zurich_self_training_dataset
        if opts.train_dataset_type == 'medium':
            func = get_medium_zurich_self_training_dataset
        mydataset = func(opts.target_data_root,stage_index=4,pseudo_label_dir=final_path,
        crop_size=opts.crop_size,use_constract=use_constract,xymap_dir = opts.xymap_dir)
    else:
        mydataset, _ = get_dataset(opts.target_dataset,opts.target_data_root,opts.crop_size,opts.test_only, plabel_dir=final_path)

    pseudo_train_loader = data.DataLoader(mydataset, batch_size=opts.batch_size, shuffle=True, num_workers=0)

    #train()
    # if not opts.continue_training:
    cur_itrs = 0 
    cur_epochs = 0
    time_cost = 0
    refine_outputs = False
    kld_con_loss_v2 = 0
    while cur_epochs < opts.epoch_one_round:
        start_kc = time.time()
        # =====  Train  =====
        model.train()
        for scheduler in schedulers:
            scheduler.step(cur_epochs)
        cur_epochs += 1
        print('###### start training in epoch {} ! ######'.format(cur_epochs))
        for opt in optimizers:
            print("当前学习率:{}".format(opt.state_dict()['param_groups'][0]['lr']))
        pack_loader,d_len = None,0
        if opts.target_dataset == 'FoggyZurich' and opts.train_dataset_type == 'medium' and  tgt_light_loader is not None:
                pack_loader = zip(cycle(source_train_loader),pseudo_train_loader,tgt_light_loader)
                d_len = min(len(source_train_dst),len(mydataset),len(tgt_light_dst)) // opts.batch_size
                print("train on medium")
        else:
            pack_loader = zip(cycle(source_train_loader),pseudo_train_loader)
            d_len = min(len(source_train_dst),len(mydataset)) // opts.batch_size
        for i,images_data in enumerate(pack_loader):
        # for i,images_data in enumerate(tqdm(pack_loader,total=d_len)):
            start_time = time.time()
            cur_itrs += 1
            off_i,off_j,isFliped = None,None,None
            soft_plabels, feat_c, xymap, closer_img = None, None, None, None
            for j in range(len(images_data)):
                _data = images_data[j]
                images = _data['images']
                labels = _data['labels']
                if 'img_paths' in _data:
                    images_path = _data['img_paths']
                if 'transform_params' in _data:
                    (off_i,off_j,isFliped) = _data['transform_params']
                if 'index' in _data:
                    img_idx = _data['index'].numpy()
                # TODO;
                if 'xymap' in _data:
                    xymap = _data['xymap']
                if 'closer_img' in _data:
                    closer_img = _data['closer_img']
                if closer_img is not None:
                    with torch.no_grad():
                        closer_img = closer_img.to(device, dtype=torch.float32)
                        # feat_c = model(closer_img,feats_only=True)
                        prob,feat_c = model(closer_img)
                        _,pred_c = torch.max(prob,dim=1)
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                for opt in optimizers:
                    opt.zero_grad()

                res = model(images, update_teacher = opts.use_teacher)
                if refine_outputs and soft_plabels is not None and j in [1,2]:
                    if j == 1:
                        dataset = mydataset
                    else:
                        dataset = tgt_light_dst

                outputs,feats = res[0], res[-1]

                pred = torch.nn.functional.interpolate(
                    outputs, size=labels.shape[1:], mode="bilinear", align_corners=False
                )
                ce_loss_v = criterion(pred, labels)
                loss = ce_loss_v
                if j != 0 and use_kld:
                    # result = torch.argmax(outputs, dim=1)
                    kld_con_loss_v = cal_kld_loss(pred, labels,255,opts.kld_weight) \

                    if kld_con_loss_v != 0.0:
                        kld_con_loss_v2 += kld_con_loss_v.detach().cpu().numpy()
                    loss += kld_con_loss_v

                interval_loss2 += ce_loss_v.detach().cpu().numpy()

                # if j != 0 and use_spatial_loss and off_i is not None:
                if use_spatial_loss and j != 0 and off_i is not None:
                    sp_results_cat = None
                    for k in range(len(images_path)):
                        sample_name = os.path.basename(images_path[k]).split('.png')[0]
                        sp_path = os.path.join(stage2.save_sp_path,'{}.npy'.format(sample_name))
                        sp_results = np.load(sp_path)
                        sp_results = sp_results[off_i[k]:off_i[k]+opts.crop_size,off_j[k]:off_j[k]+opts.crop_size]
                        if isFliped[k]:
                            sp_results = np.fliplr(sp_results)
                        h,w = feats.size()[-2:]
                        sp_results = np.expand_dims(cv2.resize(sp_results.astype(np.uint16),(w,h),cv2.INTER_NEAREST),axis=0)


                        if sp_results_cat is None:
                            sp_results_cat = sp_results
                        else:
                            sp_results_cat = np.concatenate((sp_results_cat,sp_results),axis=0)

                    sp_loss = opts.sp_weight * spatial_loss(sp_results_cat,feats)
                    loss += sp_loss
                    sp_loss_sum += sp_loss.detach().cpu().numpy()

                if xymap is not None:
                    # feats = torch.nn.functional.interpolate(
                    #     feats, size=images.shape[2:], mode="bilinear", align_corners=False
                    # )
                    # constract_loss = 0.1 * criterion_constract(feats,feats_pair,feats_pair_xy,ava_class_id)
                    #loss为什么这么大
                    constract_loss = opts.con_weight * criterion_constract(feat_c,feats,labels,xymap,pred_c,(off_i,off_j,isFliped))
                    # constract_loss = opts.con_weight * pixel_constract(feats,labels,result)
                    if constract_loss != 0:
                        loss += constract_loss
                        constract_loss_sum += constract_loss.detach().cpu().numpy()
                    del feats
        
                loss.backward()
                for opt in optimizers:
                    opt.step()

            # 输出训练信息
            if (cur_itrs) % opts.print_interval == 0:
                interval_loss2 = interval_loss2/opts.print_interval
                kld_con_loss_mean = kld_con_loss_v2/opts.print_interval
                sp_loss_mean = sp_loss_sum/opts.print_interval
                constract_loss_mean = constract_loss_sum/opts.print_interval
                print("Epoch %d, Itrs %d ,CE loss=%f, KLD contract Loss=%f constract_loss=%f sp_loss=%f time_cost=%f" %
                    (cur_epochs, cur_itrs,interval_loss2, kld_con_loss_mean, constract_loss_mean, sp_loss_mean, time_cost))
                
                sp_loss_sum = 0.0
                constract_loss_sum = 0.0
                interval_loss2 = 0.0
                time_cost = 0.0
                kld_con_loss_v2 = 0.0
            end_time = time.time()
            time_cost += round(1000 * (end_time - start_time))
            if (cur_itrs) % 100 == 0:
                # save_ckpt('checkpoints/latest_ref_2_fog_acdc_%s_%s_round_%d_epoch_%d_bs_%d.pth' %
                # save_ckpt('checkpoints/latest_%s_%s_%s_%s_%s_round_%d_epoch_%d.pth' %
                #                 (opts.target_dataset.lower(), opts.save_model_prefix, opts.model, opts.train_type, opts.train_dataset_type, round_idx, cur_epochs), opts.use_teacher)
                print("validation...")
                model.eval()
                test_score, _ = val(model,device,target_test_loader,metrics,opts)
                if test_score['Mean IoU'] > best_score:  # save best model
                    best_score = test_score['Mean IoU']
                    # save_ckpt('checkpoints/best_%s_%s_%s_%s_%s_round_%d.pth' %
                    #             (opts.target_dataset.lower(), opts.save_model_prefix, opts.model, opts.train_type, opts.train_dataset_type, round_idx), opts.use_teacher)
                print(metrics.to_str(test_score))
                print(metrics.get_results())
        print('###### End training in epoch {} !  use time:{} ######'.format(cur_epochs, time.time() - start_kc))

        
if __name__ == '__main__':
    main()
