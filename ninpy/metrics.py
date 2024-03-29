"""Collection of metrics (not loss!)."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class ConfusionMatrix(object):
    """Confusion matrix for metrics (MioU, Dice, pixel accuracy).
    Unused classes should be negative, otherwise this may include as one of scores.
    Modified from:
        https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
    """

    def __init__(self, num_classes: int) -> None:
        assert isinstance(num_classes, int)
        self.num_classes = num_classes
        # Row-wise indicates predict classes, column-wise indicates true classes.
        self._confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        assert pred.shape == true.shape
        mask = (true >= 0) & (true < self.num_classes)
        pred, true = pred[mask], true[mask]
        # Multiply `num_class` to move into the predict row of confusion matrix.
        # Add with `true` to shift to the column.
        # This comes with properties `row + classes` = diag (correct).
        confusion = pred * self.num_classes + true
        # self.num_classes ** 2 = size of a confusion matrix.
        bincount = np.bincount(confusion, minlength=self.num_classes ** 2)
        self._confusion_matrix += bincount.reshape(
            self.num_classes, self.num_classes
        )

    def pixel_accuracy(self) -> np.ndarray:
        acc = np.diag(self._confusion_matrix) / self._confusion_matrix.sum(
            axis=0
        )
        acc = np.nanmean(acc)
        return acc

    def miou_score(self) -> np.ndarray:
        # Intersection
        correct = np.diag(self._confusion_matrix)
        # Number of prediction per class.
        pred_numel = np.sum(self._confusion_matrix, axis=0)
        # Number of true per class.
        true_numel = np.sum(self._confusion_matrix, axis=1)
        iou = correct / (pred_numel + true_numel - correct)
        return np.nanmean(iou)

    def reset(self) -> None:
        self._confusion_matrix = np.zeros(self.num_classes, self.num_classes)

    def plot(self) -> None:
        sns.heatmap(self._confusion_matrix)
        plt.show()
