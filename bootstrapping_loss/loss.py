#
# Training Deep Neural Networks on Noisy Labels with Bootstrapping
# http://www-personal.umich.edu/~reedscot/bootstrap.pdf
#

import torch
from torch.nn import Module
import torch.nn.functional as F


class SoftBootstrappingLoss(Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * p) * log(p)``

    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.
        as_pseudo_label (bool): Stop gradient propagation for the term ``(1 - beta) * p``.
            Can be interpreted as pseudo-label.
    """
    def __init__(self, beta=0.95, reduce=True, as_pseudo_label=True):
        super(SoftBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce
        self.as_pseudo_label = as_pseudo_label

    def forward(self, y_pred, y):
        # cross_entropy = - t * log(p)
        beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')

        y_pred_a = y_pred.detach() if self.as_pseudo_label else y_pred
        # second term = - (1 - beta) * p * log(p)
        bootstrap = - (1.0 - self.beta) * torch.sum(F.softmax(y_pred_a, dim=1) * F.log_softmax(y_pred, dim=1), dim=1)

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap


class HardBootstrappingLoss(Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)``
    where ``z = argmax(p)``

    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.

    """
    def __init__(self, beta=0.8, reduce=True):
        super(HardBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce

    def forward(self, y_pred, y):
        # cross_entropy = - t * log(p)
        beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')

        # z = argmax(p)
        z = F.softmax(y_pred.detach(), dim=1).argmax(dim=1)
        z = z.view(-1, 1)
        bootstrap = F.log_softmax(y_pred, dim=1).gather(1, z).view(-1)
        # second term = (1 - beta) * z * log(p)
        bootstrap = - (1.0 - self.beta) * bootstrap

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap



