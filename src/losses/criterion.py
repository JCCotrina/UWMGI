import torch
import torch.nn as F
import segmentation_models_pytorch as smp
from .bce import bce 

TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
# BCELoss     = smp.losses.SoftBCEWithLogitsLoss()

def criterion(pr, gt):
    m = F.sigmoid()

    return 0.5*F.BCELoss(m(pr), gt) + 0.5*TverskyLoss(pr, gt)