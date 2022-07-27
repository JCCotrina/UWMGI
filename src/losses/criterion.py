import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()

def criterion(pr, gt):

    print("#"*20)
    print(f'tipo pr: {type(pr)}')
    print(f'tipo gt: {type(gt )}')
    return 0.5*BCELoss(pr, gt) + 0.5*TverskyLoss(pr, gt)