import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from .bce import bce 

TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()

def criterion(pr, gt):

    print(f'\n bce : {bce(pr, gt)}')
    print("#"*20)
    print(f'tipo pr: {pr.dtype}')
    print(f'tipo gt: {gt.dtype}')
    return 0.5*BCELoss(pr, gt) + 0.5*TverskyLoss(pr, gt)