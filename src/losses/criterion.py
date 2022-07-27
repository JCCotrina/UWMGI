import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()

def criterion(pr, gt):

    print(f'tipo bce: {type(BCELoss(pr, gt) )}')
    print(f'tipo bce: {type(TverskyLoss(pr, gt) )}')
    return 0.5*BCELoss(pr, gt) + 0.5*TverskyLoss(pr, gt)