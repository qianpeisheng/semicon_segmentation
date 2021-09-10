import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs_ori = inputs
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)
        # BCE = F.binary_cross_entropy(inputs, targets, reduction='mean') # unsafe to autocast
        BCE = F.binary_cross_entropy_with_logits(inputs_ori.flatten(), targets, reduction='mean') # safe to autocast

        # Dice_BCE = BCE + dice_loss

        return BCE, dice_loss
