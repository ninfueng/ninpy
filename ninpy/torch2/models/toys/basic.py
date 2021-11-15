#!usr/bin/env python3
"""
"""
import logging

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_chl: int = 1,
        num_inputs: int = 784,
        num_neurons: int = 800,
        num_hidden_layers: int = 1,
        num_classes: int = 10,
    ) -> None:
        assert isinstance(in_chl, int)
        assert isinstance(num_inputs, int)
        assert isinstance(num_neurons, int)
        assert isinstance(num_hidden_layers, int)
        assert isinstance(num_classes, int)

        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.input_layers = nn.Sequential(
            *[
                nn.Flatten(),
                nn.Linear(num_inputs, num_neurons, bias=False),
                nn.BatchNorm1d(num_neurons),
                nn.ReLU(inplace=True),
            ]
        )
        self.hidden_layers = self._make_layers(num_neurons, num_neurons)
        self.out_layers = nn.Sequential(
            *[nn.Linear(num_neurons, num_classes, bias=True)]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        o = self.input_layers(input)
        o = self.hidden_layers(o)
        o = self.out_layers(o)
        return o

    def _make_layers(self, in_features: int, out_features: int) -> nn.Module:
        layers = []
        for _ in range(self.num_hidden_layers):
            layers += [
                nn.Linear(in_features, out_features, bias=False),
                nn.BatchNorm1d(out_features),
                nn.ReLU(inplace=True),
            ]
        return nn.Sequential(*layers)


class LeNet5(nn.Module):
    """LeNet5 architecture designed for toy datasets."""

    # Can use self.ACTI_IDX and self.layers[idx] to manually change the layers.
    ACTI_IDX = [3, 7, 11, 14]
    WEIGHT_IDX = [0, 4, 9, 12, 15]

    def __init__(self, in_chl: int = 1):
        assert isinstance(in_chl, int)
        super().__init__()
        if in_chl == 1:
            NUM_FEATURES = 4
            logging.info(f"Expected MNIST-like dataset from in_chl={in_chl}.")
        elif in_chl == 3:
            NUM_FEATURES = 5
            logging.info(f"Expected CIFAR-like dataset from in_chl={in_chl}.")
        else:
            raise ValueError(f"{in_chl} may not be supported.")

        self.layers = nn.Sequential(
            *[
                nn.Conv2d(1, 6, 5, bias=False),
                nn.BatchNorm2d(6),
                nn.MaxPool2d((2, 2), stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(6, 16, 5, bias=False),
                nn.BatchNorm2d(16),
                nn.MaxPool2d((2, 2), stride=2),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(NUM_FEATURES * NUM_FEATURES * 16, 120, bias=False),
                nn.BatchNorm1d(120),
                nn.ReLU(inplace=True),
                nn.Linear(120, 84, bias=False),
                nn.BatchNorm1d(84),
                nn.ReLU(inplace=True),
                nn.Linear(84, 10, bias=True),
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        o = self.layers(input)
        return o


if __name__ == "__main__":
    input = torch.zeros(1, 1, 28, 28)
    model = LeNet5(1)
    model.eval()
    o = model(input)
    print(o.shape)
