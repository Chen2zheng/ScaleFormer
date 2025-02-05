import math
from functools import partial
import numpy as np
import torch
from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from torch import einsum, nn
from .muti_Stage import *
from .MIST import *


# __all__ = [
#     "mpvit_tiny",
#     "mpvit_xsmall",
#     "mpvit_small",
#     "mpvit_base",
# ]


def _cfg_mpvit(url="", **kwargs):
    """configuration of mpvit."""
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


class Conv2d_BN(nn.Module):
    """Convolution with BN module."""
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=None,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False)
        self.bn = norm_layer(out_ch)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity(
        )

    def forward(self, x):
        """foward function"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x

class DWCPatchEmbed(nn.Module):
    """Depthwise Convolutional Patch Embedding layer Image to Patch
    Embedding."""
    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 act_layer=nn.Hardswish):
        super().__init__()

        self.patch_conv = MBConv(
            in_channels=in_chans,
            out_channels=embed_dim,
            downscale=True if stride==2 else False,
            act_layer=act_layer,
        )
        
        # self.patch_conv = MV2Block(in_chans, embed_dim, stride)
        
    def forward(self, x):
        """foward function"""
        
        # print(x.shape)
        x = self.patch_conv(x)
        # print(x.shape)
        return x
    


class Patch_Embed_stage(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `DWCPatchEmbed` layers."""

    def __init__(self, embed_dim, num_path=4, isPool=False):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = DWCPatchEmbed(
                in_chans=embed_dim,
                embed_dim=embed_dim,
                patch_size=3,
                stride=2 if isPool else 1,
        )

    def forward(self, x):
        """foward function"""

        out = self.patch_embeds(x)
        
        return out



class MHCAEncoder(nn.Module):
    """Multi-Head Convolutional self-Attention Encoder comprised of `MHCA`
    blocks."""

    def __init__(
            self,
            dim,
            num_path=1,
            num_heads=8,
            mlp_ratio=3,
            drop_path_list=[],
            qk_scale=None,
            patchsize=[],
    ):
        super().__init__()

        self.num_path = num_path
        
        
        self.block_transformer = nn.ModuleList([ 
            ScaleformerBlock(
                in_channels=dim // num_path,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                patchsize = patchsize[idx],
                norm_layer=nn.BatchNorm2d
            ) for idx in range(num_path)
        ])
        
        self.grid_transformer = ContextFormerBlock(
            in_channels=dim,
            num_heads=num_path,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.BatchNorm2d
        )

    def forward(self, x_list, size):
        """foward function"""
        H, W = size
        res = [] 
        x_in = torch.chunk(x_list,self.num_path,dim=1)
        for x, layer in zip(x_in, self.block_transformer):

            res.append(layer(x))
            
        res = torch.cat(res,dim=1)
        
        res = self.grid_transformer(res)
        return res
    

class MHCA_stage(nn.Module):
    """Multi-Head Convolutional self-Attention stage comprised of `MHCAEncoder`
    layers."""

    def __init__(
            self,
            embed_dim,
            out_embed_dim,
            num_layers=1,
            num_heads=8,
            mlp_ratio=3,
            num_path=4,
            reverse=False,
            drop_path_list=[],
            patchsize = [],
    ):
        super().__init__()

        self.mhca_blks = nn.ModuleList([ 
            MHCAEncoder(
                embed_dim,
                num_path,
                num_heads,
                mlp_ratio,
                drop_path_list=drop_path_list,
                patchsize=patchsize,
            ) for _ in range(num_layers)
        ])
        
        
        self.num_layers = num_layers
        
        self.final = nn.Sequential(
                                    Conv2d_BN(embed_dim,out_embed_dim,act_layer=nn.GELU),
                                    # MV2Block(out_embed_dim,out_embed_dim),   
                                  )
        
    def forward(self, x_list):
        """foward function"""

        x = x_list
        _, _, H, W = x_list.shape
        for idx, layer in enumerate(self.mhca_blks):
            
            x_list = layer(x_list, size=(H, W))
        
        out = self.final(x_list)
        
        return out

def dpr_generator(drop_path_rate, num_layers, num_stages):
    """Generate drop path rate list following linear decay rule."""
    dpr_list = [
        x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))
    ]
    dpr = []
    cur = 0
    for i in range(num_stages):
        dpr_per_stage = dpr_list[cur:cur + num_layers[i]]
        dpr.append(dpr_per_stage)
        cur += num_layers[i]

    return dpr


    
# groups = 32
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
    
class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = up_conv(in_channels, out_channels)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d( out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            )
    def forward(self, x1, x2):
        x1 = self.up(x1)

        x1 = torch.cat((x2, x1), dim=1)
        x1 = self.conv_relu(x1)
        return x1
    
class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.GroupNorm(num_channels=out_ch,num_groups=groups),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.GroupNorm(num_channels=out_ch,num_groups=groups),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x
    
class conv_two(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_two, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x    
    

class MPViT_seg(nn.Module):
    """Multi-Path ViT class."""
    def __init__(
        self,
        img_size=224,
        num_stages=4,
        num_path=[4, 4, 4, 4],
        num_layers=[2, 2, 4, 1],
        embed_dims=[64, 128, 256, 512],
        mlp_ratios=[8, 8, 4, 4],
        num_heads=[8, 8, 8, 8],
        drop_path_rate=0.0,
        in_chans=3,
        num_classes=10,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_stages = num_stages
        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)

        self.stem = nn.Sequential(
            Conv2d_BN(
                in_chans,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
            Conv2d_BN(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
        )

        # Patch embeddings.
        self.patch_embed_stages = nn.ModuleList([
            Patch_Embed_stage(
                embed_dims[idx],
                num_path=1,
                isPool=False if idx == 0 else True,
            ) for idx in range(self.num_stages)
        ])

        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhca_stages = nn.ModuleList([
            MHCA_stage(
                embed_dims[idx],
                embed_dims[idx + 1]
                if not (idx + 1) == self.num_stages else embed_dims[idx],
                num_layers[idx],
                num_heads[idx],
                mlp_ratios[idx],
                num_path[idx],
                drop_path_list=dpr[idx],
                patchsize = [1,2,4,7] if idx < 2 else [1,7]
            ) for idx in range(self.num_stages)
        ])
        
        self.down1 = nn.Conv2d(in_chans, embed_dims[0]//2, kernel_size=1, stride=1, padding=0)
        self.down2 = conv_block(embed_dims[0]//2, embed_dims[0])
        self.layer4 = Decoder(embed_dims[0],embed_dims[0]*2, embed_dims[0])
        self.layer5 = Decoder(embed_dims[0], embed_dims[0],embed_dims[0]//2)

       
        self.dec_botteckneck = Block_encoder_bottleneck(embed_dims[3], embed_dims[3], 2, 0.)
        self.dec_block6 = Block_decoder(embed_dims[3], embed_dims[2], 2, 0.)
        self.dec_block7 = Block_decoder(embed_dims[2], embed_dims[1], 4, 0.)
        self.dec_block8 = Block_decoder(embed_dims[1], embed_dims[0], 4, 0.)
        self.conv1 = nn.Sequential(nn.Conv2d(embed_dims[0]//2, num_classes, kernel_size=1))
        
        self.loss1 = nn.Sequential(
                nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Upsample(scale_factor=32)
        )
        self.loss2 = nn.Sequential(
                nn.Conv2d(embed_dims[-2], num_classes, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Upsample(scale_factor=16)
        )
        self.loss3 = nn.Sequential(
                nn.Conv2d(embed_dims[-3], num_classes, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Upsample(scale_factor=8)
        )
        
        
        self.CA4 = ChannelAttention(embed_dims[0])
        self.CA3 = ChannelAttention(embed_dims[1])
        self.CA2 = ChannelAttention(embed_dims[2])
        self.CA1 = ChannelAttention(embed_dims[3])
        self.C4 = conv_two(embed_dims[0], embed_dims[0])
        self.C3 = conv_two(embed_dims[1], embed_dims[1])
        self.C2 = conv_two(embed_dims[2], embed_dims[2])
        self.C1 = conv_two(embed_dims[3], embed_dims[3])
        
        self.SA = SpatialAttention()
        
    def _init_weights(self, m):
        """initialization"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        """get classifier function"""
        return self.head

    def forward_features(self, x):
        """forward feature function"""
        
        # x's shape : [B, C, H, W]
        encoder_out = []
        x = self.stem(x)  # Shape : [B, C, H/4, W/4]

        for idx in range(self.num_stages):
            att_inputs = self.patch_embed_stages[idx](x)
            x = self.mhca_stages[idx](att_inputs)
            encoder_out.append(x)
           
            
        return encoder_out

    def forward(self, x):
        """foward function"""
        ds1 = self.down1(x)
        ds2 = self.down2(ds1)
        
        x_list = self.forward_features(x)
        x1, x2, x3, x4 = x_list
        
       
        
        x5 = self.dec_botteckneck(x4, x4)
        x5 = self.CA1(x5) * x5
        x5 = self.SA(x5) * x5
        x5 = self.C1(x5)
        loss1 = self.loss1(x5)
        # print(loss1.shape)

        x6 = self.dec_block6(x5, x3)
        x6 = self.CA2(x6) * x6
        x6 = self.SA(x6) * x6
        x6 = self.C2(x6)
        loss2 = self.loss2(x6)
        # print(loss2.shape)
        
        x7 = self.dec_block7(x6, x2)
        x7 = self.CA3(x7) * x7
        x7 = self.SA(x7) * x7
        x7 = self.C3(x7)
        loss3 = self.loss3(x7)
        # print(loss3.shape)
        
        x8 = self.dec_block8(x7, x1)
        x8 = self.CA4(x8) * x8
        x8 = self.SA(x8) * x8
        x8 = self.C4(x8)
        # print(x8.shape)
        
        d4 = self.layer4(x8, ds2)
        d5 = self.layer5(d4, ds1)
        out = self.conv1(d5)
        
        return out, loss1, loss2, loss3
        # return out, loss1


@register_model
def ScaleFormer(**kwargs):
    """mpvit_xsmall :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 2, 4, 1]
    - #channels : [64, 128, 192, 256]
    - MLP_ratio : 4
    Number of params : 10573448
    FLOPs : 2971396560
    Activations : 21983464
    """
    
    model = MPViT_seg(
        img_size=224,
        num_stages=4,
        num_path=[4, 4, 2, 2],
        num_layers=[1, 2, 4, 2],
        embed_dims=[128, 288, 384, 480],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[4, 4, 2, 2],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model




if __name__ == "__main__":
    model = ScaleFormer(num_classes=4,in_chans=1)

    model.eval()
    inputs = torch.randn(12, 1, 224, 224)

    
    with torch.no_grad():
        print(model(inputs)[0].shape)
        # print('loss1',loss1.shape)
        print(sum(p.numel() for p in model.parameters()))
        torch.save(model.state_dict(), './temp.pth')
    # print(transunet(torch.randn(1, 3, 128, 128)).shape)
#     from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

#     flops = FlopCountAnalysis(model, inputs_t )
#     param = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     acts = ActivationCountAnalysis(model, inputs_t)

  