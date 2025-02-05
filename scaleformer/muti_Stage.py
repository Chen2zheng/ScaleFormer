""" MaxViT

A PyTorch implementation of the paper: `MaxViT: Multi-Axis Vision Transformer`
    - MaxViT: Multi-Axis Vision Transformer

Copyright (c) 2021 Christoph Reich
Licensed under The MIT License [see LICENSE for details]
Written by Christoph Reich
"""
from typing import Type, Callable, Tuple, Optional, Set, List, Union
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv
from timm.models.layers import drop_path, trunc_normal_, DropPath

# from muti_block import MultiScaleAttention



class Mlp(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.,):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer()
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def _gelu_ignore_parameters(
        *args,
        **kwargs
) -> nn.Module:
    """ Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Returns:
        activation (nn.Module): GELU activation function.
    """
    activation = nn.GELU()
    return activation


class MBConv(nn.Module):
    """ MBConv block as described in: https://arxiv.org/pdf/2204.01697.pdf.

        Without downsampling:
        x ← x + Proj(SE(DWConv(Conv(Norm(x)))))

        With downsampling:
        x ← Proj(Pool2D(x)) + Proj(SE(DWConv ↓(Conv(Norm(x))))).

        Conv is a 1 X 1 convolution followed by a Batch Normalization layer and a GELU activation.
        SE is the Squeeze-Excitation layer.
        Proj is the shrink 1 X 1 convolution.

        Note: This implementation differs slightly from the original MobileNet implementation!

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downscale (bool, optional): If true downscale by a factor of two is performed. Default: False
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        drop_path (float, optional): Dropout rate to be applied during training. Default 0.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            drop_path: float = 0.,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MBConv, self).__init__()
        # Save parameter
        self.drop_path_rate: float = drop_path
        # Check parameters for downscaling
        if not downscale:
            assert in_channels == out_channels, "If downscaling is utilized input and output channels must be equal."
        # Ignore inplace parameter if GELU is used
        if act_layer == nn.GELU:
            act_layer = _gelu_ignore_parameters
        # Make main path
        self.main_path = nn.Sequential(
            norm_layer(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 4, kernel_size=(1, 1)),
            DepthwiseSeparableConv(in_chs=in_channels * 4, out_chs=in_channels * 4, stride=2 if downscale else 1,
                                   act_layer=act_layer, norm_layer=norm_layer, drop_path_rate=drop_path),
            SqueezeExcite(in_chs=in_channels * 4, rd_ratio=0.25),
            nn.Conv2d(in_channels=in_channels * 4, out_channels=out_channels, kernel_size=(1, 1))
        )
        # Make skip path
        self.skip_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        ) if downscale else nn.Identity()

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)] (downscaling is optional).
        """
        output = self.main_path(input)
        if self.drop_path_rate > 0.:
            output = drop_path(output, self.drop_path_rate, self.training)
        output = output + self.skip_path(input)
        return output


    
    
    
# class MultiScaleAttention(nn.Module):
#     """
#     Take in model size and number of heads.
#     """

#     def __init__(self, patchsize, d_model):
#         super().__init__()
#         self.patchsize = patchsize
#         self.query_embedding = nn.Conv2d(
#             d_model, d_model, kernel_size=1, padding=0
#         )
#         self.value_embedding = nn.Conv2d(
#             d_model, d_model, kernel_size=1, padding=0
#         )
#         self.key_embedding = nn.Conv2d(
#             d_model, d_model, kernel_size=1, padding=0
#         )
#         self.output_linear = nn.Sequential(
#             nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
#             nn.BatchNorm2d(d_model),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#     def forward(self, x):
#         b, c, h, w = x.size()
#         # d_k = c // len(self.patchsize)
#         output = []
#         _query = self.query_embedding(x)
#         _key = self.key_embedding(x)
#         _value = self.value_embedding(x)
#         attentions = []
#         for (width, height), query, key, value in zip(
#             self.patchsize,
#             torch.chunk(x, len(self.patchsize), dim=1),
#             torch.chunk(x, len(self.patchsize), dim=1),
#             torch.chunk(x, len(self.patchsize), dim=1),
#         ):
#             out_w, out_h = w // width, h // height

#             # 1) embedding and reshape
#             query = query.view(b, d_k, out_h, height, out_w, width)
#             query = (
#                 query.permute(0, 2, 4, 1, 3, 5)
#                 .contiguous()
#                 .view(b, out_h * out_w, d_k * height * width)
#             )
#             key = key.view(b, d_k, out_h, height, out_w, width)
#             key = (
#                 key.permute(0, 2, 4, 1, 3, 5)
#                 .contiguous()
#                 .view(b, out_h * out_w, d_k * height * width)
#             )
#             value = value.view(b, d_k, out_h, height, out_w, width)
#             value = (
#                 value.permute(0, 2, 4, 1, 3, 5)
#                 .contiguous()
#                 .view(b, out_h * out_w, d_k * height * width)
#             )

#             y, _ = attention(query, key, value)

#             # 3) "Concat" using a view and apply a final linear.
#             y = y.view(b, out_h, out_w, d_k, height, width)
#             y = y.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, d_k, h, w)
#             attentions.append(y)
#             output.append(y)

#         output = torch.cat(output, 1)
#         self_attention = self.output_linear(output)

#         return self_attention


class MultiScaleAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        b, d_k, h, w = x.size()
        # d_k = c // len(self.patchsize)
        # output = []
        query = self.query_embedding(x)
        key = self.key_embedding(x)
        value = self.value_embedding(x)
        # attentions = []
        out_w, out_h = (self.patchsize, self.patchsize)
        (width, height) = (w // out_w, h // out_h)
        
        
        query = query.view(b, d_k, out_h, height, out_w, width)
        # print(query.shape)
        query = (
            query.permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(b, out_h * out_w, d_k * height * width)
        )
        
        # print(query.shape)
        key = key.view(b, d_k, out_h, height, out_w, width)
        key = (
            key.permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(b, out_h * out_w, d_k * height * width)
        )
        value = value.view(b, d_k, out_h, height, out_w, width)
        value = (
            value.permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(b, out_h * out_w, d_k * height * width)
        )

        # y, _ = attention(query, key, value)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
                query.size(-1)
        )
        p_attn = F.softmax(scores, dim=-1)
        output = torch.matmul(p_attn, value)

        # 3) "Concat" using a view and apply a final linear.
       

            # 3) "Concat" using a view and apply a final linear.
        output = output.view(b, out_h, out_w, d_k, height, width)
        output = output.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, d_k, h, w)
        # attentions.append(y)
        # output.append(y)

        # output = torch.cat(output, 1)
        self_attention = self.output_linear(output)

        return self_attention

    
class EfficientAttention(nn.Module):
    """
        input  -> x:[B, D, H, W]
        output ->   [B, D, H, W]
    
        in_channels:    int -> Embedding Dimension 
        key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
        value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2) 
        head_count:     int -> It divides the embedding dimension by the head_count and process each part individually
        
        Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """
    
    def __init__(self, in_channels, key_channels, value_channels, head_count=2):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        # self.flag_all = flag_all
        self.keys = nn.Conv2d(in_channels, key_channels, 1) 
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(in_channels, in_channels, 1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.SE = SEBlock(in_channels=head_count, reduction_ratio=1)
        
    def forward(self, input_):
        
        input_ = self.norm(input_)  #我添加
        n, c, h, w = input_.size()
        
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        query_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            
            query = F.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
                        
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]            
            
            context = key @ value.transpose(1, 2) # dk*dv
            attended_values.append(context.unsqueeze(1))
            query_values.append(query)
        
        attended = torch.cat(attended_values, dim = 1)

        context = self.SE(attended)
        context = context.squeeze(1)
        # print("context",context.shape)
        
        for q in query_values:
            # print("before", q.shape)
            # q = (context.transpose(1, 2) @ q).reshape(q.size())
            q = (context.transpose(1, 2) @ q)
            
            # print("after", q.shape)
        
        # aggregated_values = torch.cat(attended_values, dim=1)
        
        query_values = torch.cat(query_values, dim=1)
        # print("q", query_values.shape)
        output = query_values.view(n, c, h, w)
        # print("q", query_values.shape)
        output = self.reprojection(output)

        return output
        

class SEBlock(nn.Module):
    def __init__(self, in_channels=3, reduction_ratio=1):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        # Squeeze
        out = self.avg_pool(x)
        out = out.view(out.size(0), -1)

        # Excitation
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        # Reshape and scale
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = x * out
        out = self.conv(out)
        return out    
    
    

class ScaleformerBlock(nn.Module):
    """ MaxViT Transformer block.

        Layer Normalization (LN) is applied after the grid/window partition to prevent multiple reshaping operations.
        Grid/window reverse (Unblock/Ungrid) is performed on the final output for the same reason.

    Args:
        in_channels (int): Number of input channels.
        num_heads (int, optional): Number of attention heads. Default 32
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
    """

    def __init__(
            self,
            in_channels: int,
            num_heads: int = 4,
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            patchsize: int = 7,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        """ Constructor method """
        super(ScaleformerBlock, self).__init__()
        # Save parameters
        # Init layers
        self.norm_1 = norm_layer(in_channels)
        self.attention = MultiScaleAttention(patchsize, in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(
            # in_features=in_channels,
            # hidden_features=int(mlp_ratio * in_channels),
            # act_layer=act_layer,
            # drop=drop,
            # use_conv=True,
            in_channels = in_channels,
            hidden_channels = int(mlp_ratio * in_channels),
        )
        self.patchsize = patchsize

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)].
        """
        # Save original shape
        B, C, H, W = input.shape
        # input_list = torch.chunk(input, len(self.patchsize), dim=1)
        
        # print("there ",input.shape)
        output = input + self.drop_path(self.attention(self.norm_1(input)))
        # print("here", output.shape)
        
        # Perform normalization, MLP, and dropout
        output = output + self.drop_path(self.mlp(self.norm_2(output)))

        return output

    

class ContextFormerBlock(nn.Module):
    """ MaxViT Transformer block.

        Layer Normalization (LN) is applied after the grid/window partition to prevent multiple reshaping operations.
        Grid/window reverse (Unblock/Ungrid) is performed on the final output for the same reason.

    Args:
        in_channels (int): Number of input channels.
        partition_function (Callable): Partition function to be utilized (grid or window partition).
        reverse_function (Callable): Reverse function to be utilized  (grid or window reverse).
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
    """

    def __init__(
            self,
            in_channels: int,
            num_heads: int = 4,
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        """ Constructor method """
        super(ContextFormerBlock, self).__init__()
        # Save parameters
        # Init layers
        self.norm_1 = norm_layer(in_channels)
                           # in_channels, key_channels, value_channels, head_count=4
        self.attention = EfficientAttention(in_channels, in_channels, in_channels, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(
            # in_features=in_channels,
            # hidden_features=int(mlp_ratio * in_channels),
            # act_layer=act_layer,
            # drop=drop,
            # use_conv=True,
            in_channels = in_channels,
            hidden_channels = int(mlp_ratio * in_channels),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)].
        """
        B, C, H, W = input.shape


        output = input + self.drop_path(self.attention(self.norm_1(input)))

        output = output + self.drop_path(self.mlp(self.norm_2(output)))

        return output
    
