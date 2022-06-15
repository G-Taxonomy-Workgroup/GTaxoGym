import torch.nn as nn

from torch_geometric.graphgym.register import register_loss
from torch_geometric.graphgym.config import cfg


@register_loss('multilabel_cross_entropy')
def multilabel_cross_entropy(pred, true):
    """Multilabel cross entropy loss.
    """
    if cfg.dataset.task_type == 'classification_multilabel':
        if cfg.model.loss_fun != 'cross_entropy':
            raise ValueError("Only 'cross_entropy' loss_fun supported with "
                             "'classification_multilabel' task_type.")
        if not cfg.dataset.name.startswith('ogbg-'):
            bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
            return bce_loss(pred, true.float()), pred
        else:
            bce_loss = nn.BCEWithLogitsLoss()
            is_labeled = true == true  # Filter our nans.
            return bce_loss(pred[is_labeled], true[is_labeled].float()), pred
