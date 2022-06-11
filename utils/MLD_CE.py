import torch
import torch.nn.functional as F
def reg_loss_calc(pred, label,IGNORE_LABEL, num_class,mr_weight_kld):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    valid_num = torch.sum(label != IGNORE_LABEL).float()

    softmax = F.softmax(pred, dim=1)   # compute the softmax values
    logsoftmax = F.log_softmax(pred,dim=1)   # compute the log of softmax values

    label_expand = torch.unsqueeze(label, 1).repeat(1,int(num_class),1,1)
    labels = label_expand.clone()
    labels[labels != IGNORE_LABEL] = 1.0
    labels[labels == IGNORE_LABEL] = 0.0
    labels_valid = labels.clone()
    # labels = torch.unsqueeze(labels, 1).repeat(1,num_class,1,1)
    labels = torch.cumsum(labels, dim=1)
    labels[labels != label_expand + 1] = 0.0
    del label_expand
    labels[labels != 0 ] = 1.0
    ### check the vectorized labels
    # check_labels = torch.argmax(labels, dim=1)
    # label[label == 255] = 0
    # print(torch.sum(check_labels.float() - label))
    ce = torch.sum( -logsoftmax*labels ) # cross-entropy loss with vector-form softmax
    softmax_val = softmax*labels_valid
    logsoftmax_val = logsoftmax*labels_valid
    kld = torch.sum( -logsoftmax_val/num_class )

    reg_ce = ce/valid_num + (mr_weight_kld*kld)/valid_num

    return reg_ce

def cal_kld_loss(pred, target_label,IGNORE_LABEL,mr_weight_kld):
    # defaut reg_weight
    # a = torch.sum(target_label!=IGNORE_LABEL)
    if torch.sum(target_label!=IGNORE_LABEL) == 0:
        return 0
    reg_val_matrix = torch.ones_like(target_label).type_as(pred)
    reg_val_matrix[target_label==IGNORE_LABEL]=0
    reg_val_matrix = reg_val_matrix.unsqueeze(dim=1)
    reg_ignore_matrix = 1 - reg_val_matrix
    reg_weight = torch.ones_like(pred)
    reg_weight_val = reg_weight * reg_val_matrix
    reg_weight_ignore = reg_weight * reg_ignore_matrix
    del reg_ignore_matrix, reg_weight, reg_val_matrix


    kld_reg_loss = kldloss(pred, reg_weight_val)
    kld_reg_loss =  kld_reg_loss * mr_weight_kld
    return kld_reg_loss

def kldloss(logits, weight):
    """
    logits:     N * C * H * W 
    weight:     N * 1 * H * W
    """
    val_num = weight[weight>0].numel()
    logits_log_softmax = torch.log_softmax(logits, dim=1)
    num_classes = logits.size()[1]
    kld = - 1/num_classes * weight * logits_log_softmax
    kld_reg = torch.sum(kld) / val_num
    return kld_reg