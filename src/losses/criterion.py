import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from .dice import dice 

TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()

def criterion(pr, gt):

    prin(f'\n bce : {bce(pr, gt)}')
    print("#"*20)
    print(f'tipo pr: {pr.dtype}')
    print(f'tipo gt: {gt.dtype}')
    return 0.5*BCELoss(pr, gt) + 0.5*TverskyLoss(pr, gt)