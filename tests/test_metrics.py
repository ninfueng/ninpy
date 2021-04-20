import numpy as np

from ninpy.metrics import ConfusionMatrix


class TestConfusionMatrix:
    def test_all_correct(self):
        matrix = ConfusionMatrix(3)
        matrix.update(np.arange(3), np.arange(3))
        miou = matrix.miou_score()
        assert miou == 1.0

    def test_incorrect(self):
        matrix = ConfusionMatrix(3)
        matrix.update(np.array((0, 1, 2)), np.array((0, 1, 0)))
        miou = matrix.miou_score()
        assert miou == 0.5

    def test_unknown(self):
        matrix = ConfusionMatrix(3)
        pred = np.array([0, 1, 2, 3])
        matrix.update(pred, np.arange(4))
        miou = matrix.miou_score()
        assert miou == 1.0
