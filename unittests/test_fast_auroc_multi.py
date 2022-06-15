import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from gtaxogym.metric import _fast_auroc_multi, _classidx_to_onehot


class TestFastAUROC(unittest.TestCase):
    """Test the numerical acuracy of fast_auroc_multi against sklearn.

    Test both the multiclass and multilabel settings. Note that instead of
    directly comparing the final aggregated auroc score, the comparisons are
    for each individual class.

    A small tolerance ``self.tol``  is set for maximum allowed difference
    between the auroc scores computed by ``fast_auroc_multi`` and ``sklearn``.
    A slight difference is expected due to the different integration schemes.
    Sklearn uses trapezoidal rule while fast auroc uses right Riemann sum.
    
    """

    @classmethod
    def setUpClass(cls):
        """Setup data for test cases.
        """

        cls.tol = 1e-12
        cls.seed = 0
        cls.num_samples = 10000
        cls.num_classes = 200
        np.random.seed(cls.seed)

        # randomly generate multiclass labels
        cls.true_mc = np.random.randint(cls.num_classes, size=cls.num_samples)

        # randomly generate multilabel labels
        cls.true_ml = np.random.random((cls.num_samples, cls.num_classes)) > 0.8

        # randomly generate prediction scores
        cls.pred = np.random.random((cls.num_samples, cls.num_classes))

    def compare_scores(self, scores1, scores2):
        """Compare two lists of scores and raise error if the difference
        between any pair of scores differ more than the tolerance.
        """
        for i, j in zip(scores1, scores2):
            self.assertLessEqual(np.abs(i - j), self.tol)

    def test_multiclass(self):
        """Test fast auroc in the multiclass setting.
        """
        true = self.true_mc
        pred = self.pred
        num_classes = self.num_classes

        skl_auroc_scores = [
            roc_auc_score(true == i, pred[:, i]) for i in range(num_classes)
        ]
        true_onehot = _classidx_to_onehot(true, num_classes)
        new_auroc_scores = _fast_auroc_multi(true_onehot, pred, 1)
        self.compare_scores(skl_auroc_scores, new_auroc_scores)

    def test_multilabel(self):
        """Test fast auroc in the multilabel setting.
        """
        true = self.true_ml
        pred = self.pred
        num_classes = self.num_classes

        skl_auroc_scores = [
            roc_auc_score(true[:, i], pred[:, i]) for i in range(num_classes)
        ]
        new_auroc_scores = _fast_auroc_multi(true, pred, 1)
        self.compare_scores(skl_auroc_scores, new_auroc_scores)


if __name__ == '__main__':
    unittest.main()
