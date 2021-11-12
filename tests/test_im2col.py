import torch
import torch.nn as nn
from ninpy.torch2.hw import (
    reshape_im2col_activation,
    reshape_im2col_weight,
    set_unfold,
)


def test_im2col():
    input = torch.rand(1, 3, 224, 224)
    conv = nn.Conv2d(3, 64, 3, bias=False)
    unfold = set_unfold(conv)
    with torch.no_grad():
        ref = conv(input)
        print(f"Reference shape after conv2d: {ref.shape}")
        weight2 = reshape_im2col_weight(conv)

        print(f"Shape of weights for im2col: {weight2.shape}")
        input2 = unfold(input)
        res = torch.matmul(weight2, input2)
        print(f"Shape of output after matmul: {res.shape}")
        res = reshape_im2col_activation(conv, res)
        print(f"Shape of output after reshape: {res.shape}")
    torch.testing.assert_allclose(ref, res)
