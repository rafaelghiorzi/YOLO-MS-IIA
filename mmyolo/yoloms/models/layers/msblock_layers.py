# Copyright (c) VCIP-NKU. All rights reserved.

from typing import Sequence, Union, Dict, Optional, Tuple
from mmdet.utils import OptConfigType

import warnings
from mmcv.cnn import ConvModule
from ..utils import autopad
import torch.nn as nn
from torch import Tensor

from mmengine.model import constant_init, kaiming_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm, _InstanceNorm
import torch

from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.cnn.bricks.norm import build_norm_layer
from mmcv.cnn.bricks.padding import build_padding_layer


class DWConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: Union[bool, str] = "auto",
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Optional[Dict] = dict(type="ReLU"),
        inplace: bool = True,
        with_spectral_norm: bool = False,
        padding_mode: str = "zeros",
        order: tuple = ("conv", "norm", "act"),
    ):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ["zeros", "circular"]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {"conv", "norm", "act"}

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None

        # If the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)

        # Reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding

        # Build convolution layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # Build normalization layers
        if self.with_norm:
            norm_channels = out_channels if order.index("norm") > order.index("conv") else in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)  # type: ignore
            self.add_module(self.norm_name, norm)
            if self.with_bias and isinstance(norm, (_BatchNorm, _InstanceNorm)):
                warnings.warn("Unnecessary conv bias before batch/instance norm")
        else:
            self.norm_name = None  # type: ignore

        # Build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()  # type: ignore
            if act_cfg_["type"] not in ["Tanh", "PReLU", "Sigmoid", "HSigmoid", "Swish", "GELU"]:
                act_cfg_.setdefault("inplace", inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use MSRA init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        return None

    def init_weights(self):
        if not hasattr(self.conv, "init_weights"):
            if self.with_activation and self.act_cfg["type"] == "LeakyReLU":
                nonlinearity = "leaky_relu"
                a = self.act_cfg.get("negative_slope", 0.01)
            else:
                nonlinearity = "relu"
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x: torch.Tensor, activate: bool = True, norm: bool = True) -> torch.Tensor:
        for layer in self.order:
            if layer == "conv":
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x


class MSBlock_kxk_Layer(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Union[int, Sequence[int]],
        conv_group="auto",
        conv_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
    ) -> None:
        super().__init__()
        groups = 1 if conv_group != "auto" else in_channel
        self.mid_conv = ConvModule(
            in_channel,
            in_channel,
            kernel_size,
            padding=autopad(kernel_size),
            groups=groups,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mid_conv(x)


class MSBlockBottleNeckLayer(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Union[int, Sequence[int]],
        conv_group="auto",
        conv_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
    ) -> None:
        super().__init__()
        groups = 1 if conv_group != "auto" else out_channel
        self.in_conv = ConvModule(
            in_channel, out_channel, 1, conv_cfg=conv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg
        )
        self.mid_conv = ConvModule(
            out_channel,
            out_channel,
            kernel_size,
            padding=autopad(kernel_size),
            groups=groups,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
        )
        self.out_conv = ConvModule(
            out_channel, in_channel, 1, conv_cfg=conv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)
        x = self.mid_conv(x)
        x = self.out_conv(x)
        return x


class MSBlock_kxk_1x1_Layer(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Union[int, Sequence[int]],
        conv_group="auto",
        conv_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
    ) -> None:
        super().__init__()
        groups = 1 if conv_group != "auto" else in_channel
        self.in_conv = ConvModule(
            in_channel,
            out_channel,
            kernel_size,
            padding=autopad(kernel_size),
            groups=groups,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
        )
        self.out_conv = ConvModule(
            out_channel, in_channel, 1, conv_cfg=conv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)
        x = self.out_conv(x)
        return x


class MSBlock_1x1_kxk_Layer(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Union[int, Sequence[int]],
        conv_group="auto",
        conv_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
    ) -> None:
        super().__init__()
        groups = 1 if conv_group != "auto" else in_channel
        self.in_conv = ConvModule(
            in_channel, out_channel, 1, conv_cfg=conv_cfg, act_cfg=act_cfg, norm_cfg=norm_cfg
        )
        self.out_conv = ConvModule(
            out_channel,
            in_channel,
            kernel_size,
            padding=autopad(kernel_size),
            groups=groups,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)
        x = self.out_conv(x)
        return x