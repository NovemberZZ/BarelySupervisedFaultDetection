import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable


def dice_loss_weight(score, target, mask):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target * mask)
    y_sum = torch.sum(target * target * mask)
    z_sum = torch.sum(score * score * mask)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def wce(logits, target, weights, batch_size, H, W, D):
    # Calculate log probabilities
    logp = F.log_softmax(logits, dim=1)
    # Gather log probabilities with respect to target
    logp = logp.gather(1, target.view(batch_size, 1, H, W, D))
    # Multiply with weights
    weighted_logp = (logp * weights).view(batch_size, -1)
    # Rescale so that loss is in approx. same interval
    # weighted_loss = weighted_logp.sum(1) / weights.view(batch_size, -1).sum(1)
    weighted_loss = (weighted_logp.sum(1) - 0.00001) / (weights.view(batch_size, -1).sum(1) + 0.00001)
    # Average over mini-batch
    weighted_loss = -1.0 * weighted_loss.mean()
    return weighted_loss
