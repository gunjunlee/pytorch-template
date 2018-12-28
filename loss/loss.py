import torch
import torch.nn as nn


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()

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


class WeightedBCE(nn.Module):
    def __init__(self, kernel_size=41):
        super(WeightedBCE, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, pred, target):
        batch_size, _, H, W = target.size()
        
        a = F.avg_pool2d(target, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
        ind = a.ge(0.01) * a.le(0.99)
        ind = ind.float()
        weights = torch.autograd.Variable(torch.ones(a.size())).cuda()
        w0 = weights.sum()
        weights = weights + ind*2
        w1 = weights.sum()
        weights = weights/w1*w0


        dice_loss = self.dice_loss(pred, target)
        bce_loss = nn.BCEWithLogitsLoss(weight=weights)(pred, target)
        return bce_loss


def get_loss(config):
    funcs = {

    }

    name = config.LOSS.NAME
    
    if name in funcs:
        func = funcs[name]
    else:
        func = globals()[name]
    
    print('get loss: {}'.format(name))
    return func(config)

