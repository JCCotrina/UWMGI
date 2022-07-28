import torch
import torch.nn.functional as F

def bce(pr, gt):
    #input (torch.Tensor): input data tensor with shape :math:`(B, *)`.
    #target (torch.Tensor): the target tensor with shape :math:`(B, *)`.
    print("\n","#"*12)
    print(f'input:{pr.shape}')
    print(f'target:{gt.shape}')
    print(pr)
    print(gt)
    return F.binary_cross_entropy_with_logits(pr, gt)