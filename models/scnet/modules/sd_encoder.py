from typing import List, Tuple

import torch
import torch.nn as nn

from models.scnet.utils import create_intervals, split_layer

class Downsample(nn.Module):
    """
    Downsample class implements a module for downsampling input tensors using 2D convolution.

    Args:
    - input_dim (int): Dimensionality of the input channels.
    - output_dim (int): Dimensionality of the output channels.
    - stride (int): Stride value for the convolution operation.

    Shapes:
    - Input: (B, C_in, F, T) where
        B is batch size,
        C_in is the number of input channels,
        F is the frequency dimension,
        T is the time dimension.
    - Output: (B, C_out, F // stride, T) where
        B is batch size,
        C_out is the number of output channels,
        F // stride is the downsampled frequency dimension.

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
    ):
        """
        Initializes Downsample with input dimension, output dimension, and stride.
        """
        super().__init__()
        self.conv = MDConv(output_dim, input_dim, (stride, 1), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the Downsample module.

        Args:
        - x (torch.Tensor): Input tensor of shape (B, C_in, F, T).

        Returns:
        - torch.Tensor: Downsampled tensor of shape (B, C_out, F // stride, T).
        """
        return self.conv(x)

class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
        pointwise_activation (str): name of the activation function to be used for the pointwise conv.
            Note that Conformer uses a special key `glu_` which is treated as the original default from
            the paper.
    """

    def __init__(
        self, input_dim, d_model, kernel_size, norm_type='group_norm', conv_context_size=None, pointwise_activation='glu_' # d_model is hidden dimensions
    ):
        super(ConformerConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.input_dim = input_dim

        if conv_context_size is None:
            conv_context_size = (kernel_size - 1) // 2

        self.pointwise_activation = pointwise_activation
        dw_conv_input_dim = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model * 2, kernel_size=kernel_size, stride=1, padding=0, bias=True
        )

        self.depthwise_conv = MDConv(
            in_channels=dw_conv_input_dim,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
        )

        num_groups = int(norm_type.replace("group_norm", ""))
        self.batch_norm = nn.GroupNorm(num_groups=num_groups, num_channels=d_model)

        self.activation = 'silu'
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=dw_conv_input_dim, out_channels=d_model, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, B, F, T, C, pad_mask=None, cache=None):
        nn.GELU(x) # Activate 
        x = x.reshape((B * F), T, C) # Reshape
        self.pointwise_conv1() # Cast to fp32 & Convolute

        nn.transpose(1, 2)
        x = self.pointwise_conv1 # Group Normalize #1

        x = nn.functional.glu(x, dim=1) # GLU Activation
        self.depthwise_conv # Depthwise Conv1D 

        x = x.transpose(1, 2)
        nn.GroupNorm(num_groups=1, num_channels=self.input_dim) # Group Normalize #2
        
        nn.SiLU # SiLU Activation
        self.pointwise_conv2() # Cast back to fp16 and convolute

        x = x.transpose(1, 2)
        if cache is None:
            return x
        else:
            return x, cache

    def reset_parameters_conv(self):
        pw1_max = pw2_max = self.d_model ** -0.5
        dw_max = self.kernel_size ** -0.5

        with torch.no_grad():
            nn.init.uniform_(self.pointwise_conv1.weight, -pw1_max, pw1_max)
            nn.init.uniform_(self.pointwise_conv1.bias, -pw1_max, pw1_max)
            nn.init.uniform_(self.pointwise_conv2.weight, -pw2_max, pw2_max)
            nn.init.uniform_(self.pointwise_conv2.bias, -pw2_max, pw2_max)
            nn.init.uniform_(self.depthwise_conv.weight, -dw_max, dw_max)
            nn.init.uniform_(self.depthwise_conv.bias, -dw_max, dw_max)

class MDConv(nn.Module):
    def __init__(self, out_chan, n_chunks, stride=1, bias=False):
        super(MDConv, self).__init__()
        # split_layer finds ideal kernel size by doing output/input and appending it n_chunks times
        # Depth/channel value is found by: total_channels - sum(split)
        self.split_out_channels = split_layer(out_chan, n_chunks)
        self.layers = nn.ModuleList()
        for idx in range(n_chunks):
            kernel_size = 2 * idx + 3
            padding = padding = (kernel_size - 1) // 2
            in_chan = self.split_out_channels[idx]
            conv = nn.Conv2d(in_chan, in_chan, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
            self.layers.append(conv)
        
    def forward(self, x):
        split = torch.split(x, self.split_out_channels, dim=1)
        out = torch.cat([layer(s) for layer, s in zip(self.layers, split)], dim=1)
        return out
    
class SDLayer(nn.Module):
    """
    SDLayer class implements a subband decomposition layer with downsampling and convolutional modules.

    Args:
    - subband_interval (Tuple[float, float]): Tuple representing the frequency interval for subband decomposition.
    - input_dim (int): Dimensionality of the input channels.
    - output_dim (int): Dimensionality of the output channels after downsampling.
    - downsample_stride (int): Stride value for the downsampling operation.
    - n_conv_modules (int): Number of convolutional modules.
    - kernel_sizes (List[int]): List of kernel sizes for the convolutional layers.
    - bias (bool, optional): If True, adds a learnable bias to the convolutional layers. Default is True.

    Shapes:
    - Input: (B, Fi, T, Ci) where
        B is batch size,
        Fi is the number of input subbands,
        T is sequence length, and
        Ci is the number of input channels.
    - Output: (B, Fi+1, T, Ci+1) where
        B is batch size,
        Fi+1 is the number of output subbands,
        T is sequence length,
        Ci+1 is the number of output channels.
    """

    def __init__(
        self,
        subband_interval: Tuple[float, float],
        input_dim: int,
        output_dim: int,
        downsample_stride: int,
        n_conv_modules: int,
        kernel_sizes: List[int],
        bias: bool = True,
    ):
        """
        Initializes SDLayer with subband interval, input dimension,
        output dimension, downsample stride, number of convolutional modules, kernel sizes, and bias.
        """
        super().__init__()
        self.subband_interval = subband_interval
        self.downsample = Downsample(input_dim, kernel_sizes, downsample_stride)
        self.activation = nn.GELU()
        conv_modules = [
            ConvolutionModule(
                input_dim=output_dim,
                hidden_dim=output_dim // 4,
                kernel_sizes=kernel_sizes,
                bias=bias,
            )
            for _ in range(n_conv_modules)
        ]
        self.conv_modules = nn.Sequential(*conv_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the SDLayer.

        Args:
        - x (torch.Tensor): Input tensor of shape (B, Fi, T, Ci).

        Returns:
        - torch.Tensor: Output tensor of shape (B, Fi+1, T, Ci+1).
        """
        # Downsample
        B, F, T, C = x.shape
        x = x[:, int(self.subband_interval[0] * F) : int(self.subband_interval[1] * F)]
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        x = self.activation(x)
        x = x.permute(0, 2, 3, 1)
        B, F, T, C = x.shape
        # Convolute
        x = self.conv_modules(B, F, T, C) # Returned convoluted tensor
        x = x.reshape(B, F, T, C)

        return x


class SDBlock(nn.Module):
    """
    SDBlock class implements a block with subband decomposition layers and global convolution.

    Args:
    - input_dim (int): Dimensionality of the input channels.
    - output_dim (int): Dimensionality of the output channels.
    - bandsplit_ratios (List[float]): List of ratios for splitting the frequency bands.
    - downsample_strides (List[int]): List of stride values for downsampling in each subband layer.
    - n_conv_modules (List[int]): List specifying the number of convolutional modules in each subband layer.
    - kernel_sizes (List[int], optional): List of kernel sizes for the convolutional layers. Default is None.

    Shapes:
    - Input: (B, Fi, T, Ci) where
        B is batch size,
        Fi is the number of input subbands,
        T is sequence length,
        Ci is the number of input channels.
    - Output: (B, Fi+1, T, Ci+1) where
        B is batch size,
        Fi+1 is the number of output subbands,
        T is sequence length,
        Ci+1 is the number of output channels.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bandsplit_ratios: List[float],
        downsample_strides: List[int],
        n_conv_modules: List[int],
        kernel_sizes: List[int] = None,
    ):
        """
        Initializes SDBlock with input dimension, output dimension, band split ratios, downsample strides, number of convolutional modules, and kernel sizes.
        """
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 1]
        assert sum(bandsplit_ratios) == 1, "The split ratios must sum up to 1."
        subband_intervals = create_intervals(bandsplit_ratios)
        self.sd_layers = nn.ModuleList(
            SDLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                subband_interval=sbi,
                downsample_stride=dss,
                n_conv_modules=ncm,
                kernel_sizes=kernel_sizes,
            )
            for sbi, dss, ncm in zip(
                subband_intervals, downsample_strides, n_conv_modules
            )
        )
        self.global_conv2d = nn.Conv2d(output_dim, output_dim, 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs forward pass through the SDBlock.

        Args:
        - x (torch.Tensor): Input tensor of shape (B, Fi, T, Ci).

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Output tensor and skip connection tensor.
        """
        x_skip = torch.concat([layer(x) for layer in self.sd_layers], dim=1)
        print(x.shape)
        x = self.global_conv2d(x_skip.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        print(x.shape)
        return x, x_skip
