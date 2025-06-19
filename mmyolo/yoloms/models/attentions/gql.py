# Copyright (c) VCIP-NKU. All rights reserved.

import torch
from torch import nn

from mmyolo.registry import MODELS


@MODELS.register_module()
class GQL(nn.Module):
    """Global Query Layer (GQL).

    This module implements a global query attention mechanism. It uses an adaptive average pooling layer 
    to reduce the spatial dimensions of the input tensor, followed by a convolutional layer to compute 
    attention weights based on a query tensor.

    Args:
        dim (int): The number of input channels.
        length (int): The length of the query tensor. Defaults to 3.
        size (int): The target spatial size for adaptive average pooling. Defaults to 4.

    Forward Args:
        inputs (torch.Tensor): The input tensor of shape (B, C, H, W), where B is the batch size, 
            C is the number of channels, and H, W are the spatial dimensions.
        query (torch.Tensor): The query tensor of shape (N, D), where N is the number of queries 
            and D is the query dimension.

    Returns:
        torch.Tensor: The output tensor after applying the attention mechanism, with the same shape as `inputs`.
    """

    def __init__(self, dim: int, length: int = 3, size: int = 4) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((size, size))
        self.length = length
        self.down_conv = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GQL module.

        Args:
            inputs (torch.Tensor): The input tensor of shape (B, C, H, W).
            query (torch.Tensor): The query tensor of shape (N, D).

        Returns:
            torch.Tensor: The output tensor after applying the attention mechanism.
        """
        n, _ = query.shape
        b, C, _, _ = inputs.shape

        # Apply adaptive average pooling and convolution
        pooled = self.pool(inputs)
        outs = self.down_conv(pooled)

        # Compute attention weights
        weight = torch.matmul(query[None, ...].repeat(b, 1, 1), outs.reshape(b, -1, 1)).sigmoid()

        # Apply attention weights to the input tensor
        inputs = inputs * weight.repeat(1, C // n, 1).unsqueeze(-1)
        return inputs