"""Quantization tools.
@author: Ninnart Fuengfusin
"""
import math

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log2_quant = lambda x: 2 ** round(math.log2(x))
log2_quant_num_shift = lambda x: round(math.log2(x))
int8_quant = lambda x, s, zp: ((x / s) + zp).round().clamp(-128, 127)
uint8_quant = lambda x, s, zp: ((x / s) + zp).round().clamp(0, 255)
tqt_int8_quant = lambda x, s: (x / s).round().clamp(-128, 127)
tqt_uint8_quant = lambda x, s: (x / s).round().clamp(0, 255)
merge_scale2bn = lambda w_bn, b_bn, s_in, s_out: (
    s_in * w_bn / s_out,
    b_bn / s_out,
)


def basic_quant_config():
    """"""
    qconfig = torch.quantization.QConfig(
        activation=torch.quantization.FakeQuantize.with_args(
            observer=torch.quantization.MinMaxObserver,
            quant_min=0,
            quant_max=255,
            qscheme=torch.per_tensor_symmetric,
            dtype=torch.quint8,
            reduce_range=False,
        ),
        weight=torch.quantization.FakeQuantize.with_args(
            observer=torch.quantization.MinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False,
        ),
    )
    return qconfig


def measure_sparse(*ws) -> float:
    """Measure the sparsity of input tensors or *tensors.
    Example:
    >>> measure_sparse(w0, w1)
    """
    if not ws:
        # Detecting in case, empty tuple of ws (max pooling or others).
        sparse = torch.tensor(0.0)
    else:
        # In case, not empty tuple.
        total_sparsity, num_params = 0, 0
        for w in ws:
            if w is None:
                # In case of w is None.
                continue
            w = w.data
            device = w.device
            num_params += w.numel()
            total_sparsity += torch.where(
                w == 0.0,
                torch.tensor(1.0).to(device),
                torch.tensor(0.0).to(device),
            ).sum()
        if num_params == 0:
            # Protecting in case, all parameters is zeros. 0/0 = ZeroDivisionError.
            sparse = torch.tensor(0.0)
        else:
            sparse = total_sparsity / num_params
    return sparse.item()


def ternary_threshold(delta: float = 0.7, *ws):
    """Ternary threshold find in ws."""
    assert isinstance(delta, float)
    assert 0.0 < delta < 1.0
    num_params, sum_w = 0, 0

    if not ws:
        # In case, of all params cannot be found.
        threshold = torch.tensor(np.nan)
    else:
        for w in ws:
            w = w.data
            num_params += w.numel()
            sum_w += w.abs().sum()
        threshold = delta * (sum_w / num_params)
    return threshold


class BinConnectQuant(torch.autograd.Function):
    r"""BinaryConnect quantization.
    Refer:
        https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
        https://discuss.pytorch.org/t/difference-between-apply-an-call-for-an-autograd-function/13845/3
    """

    @staticmethod
    def forward(ctx, w):
        """Require w be in range of [0, 1].
        Otherwise, it is not in activate range.
        """
        ctx.save_for_backward(w)
        return w.sign()

    @staticmethod
    def backward(ctx, grad_o):
        r"""Clipped grad where, -1 < w < 1."""
        (w,) = ctx.saved_tensors
        device = w.device
        grad_i = grad_o.clone()

        grad_i = torch.where(w < 1, grad_i, torch.tensor(0.0).to(device))
        grad_i = torch.where(w > -1, grad_i, torch.tensor(0.0).to(device))
        return grad_i


class TerQuant(torch.autograd.Function):
    r"""Ternary Weight quantization function."""

    @staticmethod
    def forward(ctx, w, threshold):
        ctx.save_for_backward(w, threshold)
        device = w.device
        w_ter = torch.where(
            w > threshold,
            torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device),
        )
        w_ter = torch.where(
            w.abs() <= -threshold, torch.tensor(0.0).to(device), w_ter
        )
        w_ter = torch.where(
            w < -threshold, torch.tensor(-1.0).to(device), w_ter
        )
        return w_ter

    @staticmethod
    def backward(ctx, grad_o):
        r"""Back propagation using same as an identity function."""
        (
            w,
            thre,
        ) = ctx.saved_tensors
        grad_i = grad_o.clone()
        return grad_i, None


class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_q = None

    def forward(self, x, *args):
        y = F.linear(x, self.weight, self.bias)
        return y


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_q = None

    def forward(self, x, *args):
        y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        return y


class BinConnectConv2d(Conv2d):
    """BinaryConnect."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, *args):
        self.weight_q = BinConnectQuant.apply(self.weight)
        y = F.conv2d(x, self.weight_q, self.bias, self.stride, self.padding)
        return y


class BinConnectLinear(Linear):
    r"""Binarized neural networks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, *args):
        self.weight_q = BinConnectQuant.apply(self.weight)
        y = F.linear(x, self.weight_q, self.bias)
        return y


class TerLinear(Linear):
    r"""Ternary Weight Layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = 0.7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        threshold = ternary_threshold(self.delta, self.weight)
        self.weight_q = TerQuant.apply(self.weight, threshold)
        x = F.linear(x, self.weight_q, self.bias)
        return x


class TerConv2d(Conv2d):
    r"""Ternary Weight Layer.
    Example:
    >>> TerConv2d(3, 5, 3)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = 0.7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        threshold = ternary_threshold(self.delta, self.weight)
        self.weight_q = TerQuant.apply(self.weight, threshold)
        x = F.conv2d(x, self.weight_q, self.bias, self.stride, self.padding)
        return x


class QuantModule(nn.Module):
    def __init__(self):
        super().__init__()

    def get_name_layers(self):
        """Get name of layers in two groups.
        Shortcut or normal path. This is useful to analysis.
        The quantization effect of residual conenction.
        Not include nn.Identity in case of shortcut.
        """
        list_shortcut = []
        list_module = []
        for n, m in self.named_modules():
            if n.find("s") > -1:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    list_shortcut.append(n)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                list_module.append(n)
        return list_module, list_shortcut

    def quantization(
        self, module_types: str, shortcut_types: str, verbose: bool = True
    ) -> None:
        """Convert quantization type to each module."""
        list_name_modules, list_name_shortcuts = self.get_name_layers()
        assert len(shortcut_types) == len(list_name_shortcuts)
        assert len(module_types) == len(list_name_modules)
        assert isinstance(shortcut_types, str)
        assert isinstance(module_types, str)

        shortcut_types = shortcut_types.lower()
        module_types = module_types.lower()
        self._cvt2quant(list_name_shortcuts, shortcut_types, verbose)
        self._cvt2quant(list_name_modules, module_types, verbose)

    def _cvt2quant(
        self, list_module_names: list, weight_types: str, verbose: bool = True
    ) -> None:
        for w, n in zip(weight_types, list_module_names):
            l = getattr(self, n)
            if isinstance(l, nn.Conv2d):
                in_channels = l.in_channels
                out_channels = l.out_channels
                kernel_size = l.kernel_size
                stride = l.stride
                padding = l.padding
                bias = l.bias is not None

                if w == "f":
                    # Make nn.Conv2d able to accept an additional input such as threshold, and ignore it.
                    setattr(
                        self,
                        n,
                        Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                        ),
                    )
                elif w == "t":
                    setattr(
                        self,
                        n,
                        ModTerConv2d(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                        ),
                    )
                elif w == "b":
                    setattr(
                        self,
                        n,
                        BinConv2d(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                        ),
                    )
                else:
                    raise NotImplementedError(
                        f"type_ws should be in [f, t, b], your {w}"
                    )

                if verbose:
                    logging.info(f"Convert {n} layer with type {w}.")

            elif isinstance(l, nn.Linear):
                in_features = l.in_features
                out_features = l.out_features
                bias = l.bias is not None

                if w == "f":
                    setattr(
                        self, n, Linear(in_features, out_features, bias=bias)
                    )
                elif w == "t":
                    setattr(
                        self,
                        n,
                        ModTerLinear(in_features, out_features, bias=bias),
                    )
                elif w == "b":
                    setattr(
                        self,
                        n,
                        BinLinear(in_features, out_features, bias=bias),
                    )
                else:
                    raise NotImplementedError(
                        f"type_ws should be in [f, t, b], your {w}"
                    )

                if verbose:
                    logging.info(f"Convert {n} layer with type {w}.")

            else:
                if verbose:
                    logging.info(
                        f"Skipping {n} layer with quantization type {w}."
                    )

    def sparse_all(self) -> float:
        """Get all sparse from all layers that has attribute, weight_q."""
        ws = []
        for l in self.modules():
            if hasattr(l, "weight_q"):
                ws.append(l.weight_q)
        return measure_sparse(*ws)

    def sparse_layerwise(self) -> list:
        """Sparse layerwise"""
        spar_ws = []
        for l in self.modules():
            if hasattr(l, "weight_q"):
                spar_ws.append(measure_sparse(l.weight_q))
        return spar_ws


class ResNet18(QuantModule):
    """ResNet18."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        assert isinstance(in_channels, int)
        # First conv layer.
        self.l0 = nn.Conv2d(
            3,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.l1 = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.l2 = nn.ReLU(inplace=True)

        # Layer 1
        self.l3 = nn.Conv2d(
            64,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.l4 = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.l5 = nn.ReLU(inplace=True)

        self.l6 = nn.Conv2d(
            64,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.l7 = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        # Shortcut
        self.s0 = nn.Identity()
        self.l8 = nn.ReLU(inplace=True)

        self.l9 = nn.Conv2d(
            64,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.l10 = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.l11 = nn.ReLU(inplace=True)
        self.l12 = nn.Conv2d(
            64,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.l13 = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        # Shortcut
        self.s1 = nn.Identity()
        self.l14 = nn.ReLU(inplace=True)

        # Layer 2
        self.l15 = nn.Conv2d(
            64,
            128,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        self.l16 = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.l17 = nn.ReLU(inplace=True)
        self.l18 = nn.Conv2d(
            128,
            128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.l19 = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        # Shortcut
        self.s2 = nn.Conv2d(
            64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False
        )
        self.s3 = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.l20 = nn.ReLU(inplace=True)

        self.l21 = nn.Conv2d(
            128,
            128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.l22 = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.l23 = nn.ReLU(inplace=True)
        self.l24 = nn.Conv2d(
            128,
            128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.l25 = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        # Shortcut
        self.s4 = nn.Identity()
        self.l26 = nn.ReLU(inplace=True)

        # Layer 3
        self.l27 = nn.Conv2d(
            128,
            256,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        self.l28 = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.l29 = nn.ReLU(inplace=True)
        self.l30 = nn.Conv2d(
            256,
            256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.l31 = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        # Shortcut
        self.s5 = nn.Conv2d(
            128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False
        )
        self.s6 = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.l32 = nn.ReLU(inplace=True)

        self.l33 = nn.Conv2d(
            256,
            256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.l34 = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.l35 = nn.ReLU(inplace=True)
        self.l36 = nn.Conv2d(
            256,
            256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.l37 = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        # Shortcut
        self.s7 = nn.Identity()
        self.l38 = nn.ReLU(inplace=True)

        # Layer 4
        self.l39 = nn.Conv2d(
            256,
            512,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        self.l40 = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.l41 = nn.ReLU(inplace=True)
        self.l42 = nn.Conv2d(
            512,
            512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.l43 = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        # Shortcut
        self.s8 = nn.Conv2d(
            256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
        )
        self.s9 = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.l44 = nn.ReLU(inplace=True)

        self.l45 = nn.Conv2d(
            512,
            512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.l46 = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.l47 = nn.ReLU(inplace=True)
        self.l48 = nn.Conv2d(
            512,
            512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.l49 = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        # Shortcut
        self.s10 = nn.Identity()
        self.l50 = nn.ReLU(inplace=True)

        self.l51 = nn.Flatten()
        self.l52 = nn.Linear(in_features=512, out_features=10, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)

        tmp = x
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        tmp = self.s0(tmp)
        x = self.l8(x + tmp)

        tmp = x
        x = self.l9(x)
        x = self.l10(x)
        x = self.l11(x)
        x = self.l12(x)
        x = self.l13(x)
        tmp = self.s1(tmp)
        x = self.l14(x + tmp)

        tmp = x
        x = self.l15(x)
        x = self.l16(x)
        x = self.l17(x)
        x = self.l18(x)
        x = self.l19(x)
        tmp = self.s2(tmp)
        tmp = self.s3(tmp)
        x = self.l20(x + tmp)

        tmp = x
        x = self.l21(x)
        x = self.l22(x)
        x = self.l23(x)
        x = self.l24(x)
        x = self.l25(x)
        tmp = self.s4(tmp)
        x = self.l26(x + tmp)

        tmp = x
        x = self.l27(x)
        x = self.l28(x)
        x = self.l29(x)
        x = self.l30(x)
        x = self.l31(x)
        tmp = self.s5(tmp)
        tmp = self.s6(tmp)
        x = self.l32(x + tmp)

        tmp = x
        x = self.l33(x)
        x = self.l34(x)
        x = self.l35(x)
        x = self.l36(x)
        x = self.l37(x)
        tmp = self.s7(tmp)
        x = self.l38(x + tmp)

        tmp = x
        x = self.l39(x)
        x = self.l40(x)
        x = self.l41(x)
        x = self.l42(x)
        x = self.l43(x)
        tmp = self.s8(tmp)
        tmp = self.s9(tmp)
        x = self.l44(x + tmp)

        tmp = x
        x = self.l45(x)
        x = self.l46(x)
        x = self.l47(x)
        x = self.l48(x)
        x = self.l49(x)
        tmp = self.s10(tmp)
        x = self.l50(x + tmp)

        x = F.avg_pool2d(x, 4)
        x = self.l51(x)
        x = self.l52(x)
        return x


def cvt2quant(model: nn.Module, type_ws: str, verbose: bool = True) -> None:
    """Convert each weight layer, `nn.Conv2d` or `nn.Linear` to as define as `type_ws`.
    In this case, all of conv module must be declare in first layer.
    Example:
    ```
    model = TestNet() # Three layers NN.
    type_ws = 'ftb'
    cvt2quant(model, type_ws)
    ```
    """
    assert len(type_ws) == len(list(model.named_modules())[1:])
    assert isinstance(verbose, bool)
    assert isinstance(type_ws, str)
    type_ws = type_ws.lower()
    for w, (n, l) in zip(type_ws, list(model.named_modules())[1:]):
        if isinstance(l, nn.Conv2d):
            in_channels = l.in_channels
            out_channels = l.out_channels
            kernel_size = l.kernel_size
            stride = l.stride
            padding = l.padding
            bias = l.bias is not None

            if w == "f":
                # Make nn.Conv2d able to accept an additional input such as threshold, and ignore it.
                setattr(
                    model,
                    n,
                    Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        bias=bias,
                    ),
                )
            elif w == "t":
                setattr(
                    model,
                    n,
                    TerConv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        bias=bias,
                    ),
                )
            elif w == "b":
                setattr(
                    model,
                    n,
                    BinConv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        bias=bias,
                    ),
                )
            else:
                raise NotImplementedError(
                    f"type_ws should be in [f, t, b], your {type_ws}"
                )

            if verbose:
                logging.info(f"Convert {n} layer with type {type_ws}.")

        elif isinstance(l, nn.Linear):
            in_features = l.in_features
            out_features = l.out_features
            bias = l.bias is not None

            if w == "f":
                setattr(model, n, Linear(in_features, out_features, bias=bias))
            elif w == "t":
                setattr(
                    model, n, TerLinear(in_features, out_features, bias=bias)
                )
            elif w == "b":
                setattr(
                    model, n, BinLinear(in_features, out_features, bias=bias)
                )
            else:
                raise NotImplementedError(
                    f"type_ws should be in [f, t, b], your {type_ws}"
                )

            if verbose:
                logging.info(f"Convert {n} layer with type {type_ws}.")

        else:
            if verbose:
                logging.info(
                    f"Skipping {n} layer with quantization type {type_ws}."
                )


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Conv2d(
            3,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.l1 = nn.Conv2d(
            64,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.l2 = nn.Conv2d(
            128,
            128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True,
        )

    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        return x


if __name__ == "__main__":
    from functools import reduce

    model = TestNet()
    type_ws = "ftb"
    cvt2quant(model, type_ws)
    print(model)

    # model = ResNet18()
    # LayerConverter.convert_conv_layers(
    #     model, nn.Conv2d, TerConv2d, True)
    model = ResNet18()
    print(f"Number of module: {len(model._modules)}")

    test = torch.zeros(1, 3, 32, 32)
    test_out = model(test)
    print(test_out.shape)

    list_forward, list_shortcut = model.get_name_layers()
    print(list_shortcut)
    print(list_forward)
    print(len(list_forward))
    print(len(list_shortcut))

    type_modules = ["b" for i in range(18)]
    type_modules = reduce(lambda x, y: x + y, type_modules)
    type_shortcuts = ["t" for i in range(3)]
    type_shortcuts = reduce(lambda x, y: x + y, type_shortcuts)
    print(type_shortcuts)

    model.quantization(type_shortcuts, type_modules)
    print(model)
