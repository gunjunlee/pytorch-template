import torch
import torch.nn as nn

class Dice_loss(nn.Module):
    def __init__(self):
        super(Dice_loss, self).__init__()
    
    def forward(self, pred, target):
        """calc dice loss
        
        Args:
            pred (torch.tensor): (N, H, W)
            target (torch.tensor): (N, H, W)
        
        Returns:
            (torch.tensor): dice loss
        """
        pred = pred.float()
        target = target.float()
        smooth = 1e-5

        p = torch.sigmoid(pred)
        
        inter = (target*p).sum(dim=2).sum(dim=1)
        dim1 = (p).sum(dim=2).sum(dim=1)
        dim2 = (target).sum(dim=2).sum(dim=1)

        coeff = (2 * inter + smooth) / (dim1 + dim2 + smooth)
        dice_total = 1-coeff.sum(dim=0)/coeff.size(0)
        return dice_total

