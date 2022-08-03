import torch
import torch.nn.functional as F

def dice(pr, gt, eps=1e-3):
        gt = gt.to(torch.float32)
        pr = (pr>0.5).to(torch.float32)
        tp = torch.sum(gt * pr, dim=(2,3))
        fp = torch.sum(pr, axis=(2, 3)) 
        fn = torch.sum(gt, axis=(2, 3)) 
        loss = (2.*tp + eps) / (fn + fp + eps)
        return torch.mean(loss, dim=(1,0))