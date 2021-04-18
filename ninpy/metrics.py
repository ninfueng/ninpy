# miou and dice coefficient confustion matrix.
import numpy as np

class ConfusionMatrix:
    """Confusion Matrix for tracking matrix.
    Modified:
        https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        # Row predict, Column true.
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred, true):
        assert pred.shape == true.shape
        mask = (true >= 0) & (true < self.num_classes)
        pred, true = pred[mask], true[mask]
        # Multiply `num_class` to move into the predict row of confusion matrix.
        # Add with `true` to shift to the column.
        # This comes with properties `row + classes` = diag (correct).
        confusion = pred * self.num_classes + true
        bincount = np.bincount(confusion, minlength=self.num_classes**2)
        self.confusion_matrix += bincount.reshape(self.num_classes, self.num_classes)

    def pixel_accuracy(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        acc = np.nanmean(acc)
        return acc

    def miou_score(self):
        correct = np.diag(self.confusion_matrix)
        # Number of prediction per class.
        pred_numel = np.sum(self.confusion_matrix, axis=0)
        # Number of true per class.
        true_numel = np.sum(self.confusion_matrix, axis=1)
        iou = correct/(pred_numel + true_numel - correct)
        return np.nanmean(iou)

    def reset(self):
        self.confusion_matrix = np.zeros(self.num_classes, self.num_classes)
