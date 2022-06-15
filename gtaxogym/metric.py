import torch
import numpy as np
from numba import njit, bool_
from torch_geometric.graphgym.config import cfg

from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, mean_absolute_error, mean_squared_error


def auroc_multi(true, pred, min_num_pos=1):
    """AUROC scores for multiclass and multilabel tasks.

    Similar to the scikit-learn roc_auc_score with ``multi_class = 'ovr'``, but
    takes care of the cases where no positive label is availble in a specific
    split. Specifically, this function computes the average AUROC score across
    all classes exluding those with no positive example.

    Args:
        true: 1-d label index array in the case of multiclass, and n_classes-d
            one-hot encoded array in the case of multilabel task.
        pred: n_classes-d prediction score array.
        min_num_pos (int): positive integer indicating the minimum number of
            positive examples required for evaluation.

    Return:
        Averaged nontrivial AUROC scores.

    """
    score = count = 0
    for i in range(pred.shape[1]):
        if true.dim() == 2:  # multilabel
            y_true = true[:, i] == 1
        else:  # multiclass
            y_true = true == i

        # skip if no possitive example is available
        if y_true.sum() >= min_num_pos:
            y_pred = pred[:, i]
            score += roc_auc_score(y_true, y_pred)
            count += 1

    return score / count


def fast_auroc_multi(true, pred, min_num_pos=1):
    """Wrap the fast NUMBA implementation of AUROC scores computation for
    multiclass and multilabel tasks.

    Compute AUROC for each class individually. If the task is multiclass,
    then convert to the label to one-hot encoded format using the
    ``_classidx_to_onehot`` function first and treat it again as 'multilabel'.

    Args:
        true: n_classes-d one-hot encoded label array.
        pred: n_classes-d prediction score array.
        min_num_pos (int): positive integer indicating the minimum number of
            positive examples required for evaluation.

    Return:
        Averaged nontrivial AUROC scores.

    """
    n_classes = cfg.share.dim_out
    pred_np = pred.numpy()

    true_dim = true.dim()
    if true_dim == 2:  # multilabel
        true_np = true.numpy().astype(bool)
    elif true_dim == 1:  # multiclass
        true_np = _classidx_to_onehot(true.numpy(), n_classes)
    else:
        raise ValueError(
            f'True label dimension must be either 1 or 2, received {true_dim}'
        )

    scores = _fast_auroc_multi(true_np, pred_np, min_num_pos)
    score = np.nanmean(scores)

    return score


@njit(nogil=True)
def _classidx_to_onehot(y_true, n_classes):
    """Convert multiclass label array to onehot encoded label format.
    """
    y_true_onehot = np.zeros((y_true.size, n_classes), dtype=bool_)
    for i, j in enumerate(y_true):
        y_true_onehot[i, j] = True
    return y_true_onehot


@njit(nogil=True)
def _fast_auroc_multi(y_true, y_pred, min_num_pos):
    """NUMBA implementation for computing AUROC scores in multilabel settings.

    Iterate over individual class and compute the corresponding AUROC by
    1. Sorting label vector based on the corresponding predictions.
    2. Scaning through the sorted label vector and compute the true positive
        rate (``tpr``) and false positive rate (``fpr``).
    3. Integrating over the ROC curve (``tpr`` vs ``fpr``) via right Riemann
        sum.

    Note:
        If a class does not have any available positive example, then set the
            corresponding AUROC score to ``np.nan``

    Args:
        y_true (numpy.ndarray): n-classes-d one-hot encoded label array.
        y_pred (numpy.ndarray): n_classes-d prediction score array.
        min_num_pos (int): positive intger indicating the minimum number of
            positive examples required for evaluation.

    Return:
        1-d array of size n_classes containing auroc scores for the
            corresponding classes (nan if no positive example for a class).

    """
    n, n_classes = y_true.shape
    auroc_scores = np.zeros(n_classes)

    # Iterate over classes and compute AUROC individually
    for j in range(n_classes):
        n_pos = np.count_nonzero(y_true[:, j])
        n_neg = n - n_pos

        # Skip computation if insufficient positive example is available
        if n_pos < min_num_pos:
            auroc_scores[j] = np.nan
        else:
            # Initializing ROC and sort the labels by predictions
            auroc_score = n_tp = n_fp = tpr = fpr = 0
            y_true_sorted = y_true[y_pred[:, j].argsort()[::-1], j]

            # Iterate over the sorted prediction array and integrate ROC
            for i in range(n):
                prev_tpr, prev_fpr = tpr, fpr

                if y_true_sorted[i]:
                    n_tp += 1
                else:
                    n_fp += 1

                tpr = n_tp / n_pos
                fpr = n_fp / n_neg

                interval = fpr - prev_fpr
                if interval > 0:
                    auroc_score += tpr * interval  # right Riemann sum

            auroc_scores[j] = auroc_score

    return auroc_scores
