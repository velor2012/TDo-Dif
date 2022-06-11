import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np
class SpatialLoss(nn.Module):
    def __init__(self, ignore_index=255):
        '''
        description: 
        param {*} self
        param {*} superpixels_results
        param {*} ignore_index
        return {*}
        '''
        super(SpatialLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, superpixels_results, feats):
        var_loss = 0
        bs_nums = len(feats)
        c = 0
        for bs in range(bs_nums):
            sp_ids = np.unique(superpixels_results[bs])
            for id in sp_ids:
                c += 1
                mask = superpixels_results[bs] == id
                if np.sum(mask) < 2:
                    continue
                ava_feats = feats[bs,:,mask]
                mean_f = torch.mean(ava_feats,dim=1).unsqueeze(dim=1)
                # print(mean_f.size())
                # print(ava_feats.size())
                var_loss_t = torch.pow(ava_feats - mean_f,2)
                # var_loss_t =torch.var(ava_feats,dim=1)
                var_loss += var_loss_t.mean()
            return var_loss/c