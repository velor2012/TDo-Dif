import torch
import torch.nn.functional as F
import numpy as np

class SCELoss(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=1,ignore_index=255, num_classes=19):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    # https://github.com/microsoft/ProDA/blob/9ba80c7dbbd23ba1a126e3f4003a72f27d121a1f/models/adaptation_modelv2.py#L28
    def rce(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        mask = (labels != self.ignore_index).float()
        labels[labels==self.ignore_index] = self.num_classes
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes + 1).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot.permute(0,3,1,2)[:,:-1,:,:], min=1e-4, max=1.0)
        rce = -(torch.sum(pred * torch.log(label_one_hot), dim=1) * mask).sum() / (mask.sum() + 1e-6)
        return rce

    def forward(self, pred, labels):

        # CCE
        ce = self.cross_entropy(pred, labels)
        a  = np.unique(labels.clone().detach().cpu().numpy())
        # RCE
        rce = self.rce(pred,labels.clone())

        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss