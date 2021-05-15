import torch.nn as nn
from torchvision.models import resnet18

from ninpy.torch2 import add_weight_decay


class TestAddWeightDecay:
    def test_conv(self):
        conv = nn.Conv2d(1, 1, 1)
        conv2 = nn.ConvTranspose2d(1, 1, 1)
        sequence = nn.Sequential(conv, conv2)
        not_decay, decay = add_weight_decay(sequence, 9.0)

        assert len(not_decay["params"]) == 2
        assert len(decay["params"]) == 2
        assert not_decay["weight_decay"] == 0.0
        assert decay["weight_decay"] == 9.0

    def test_batchnorm(self):
        batch = nn.BatchNorm2d(1)
        sync = nn.SyncBatchNorm(1)
        instance = nn.SyncBatchNorm(1)
        sequence = nn.Sequential(batch, sync, instance)
        not_decay, decay = add_weight_decay(sequence, 9.0)

        # Both weights and biases of BatchNorm go to weight decay.
        assert len(not_decay["params"]) == 6
        assert len(decay["params"]) == 0

    def test_linear(self):
        linear = nn.Linear(1, 1)
        not_decay, decay = add_weight_decay(linear, 9.0)

        assert len(not_decay["params"]) == 1
        assert len(decay["params"]) == 1

    def test_resnet18(self):
        model = resnet18(pretrained=False)
        not_decay, decay = add_weight_decay(model, 9.0)
        print(len(not_decay["params"]))
        # conv + linear only weights = 21
        assert len(decay["params"]) == 21
        assert len(not_decay["params"]) == 41
