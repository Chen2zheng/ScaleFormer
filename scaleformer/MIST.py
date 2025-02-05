import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn as nn



class Attention(nn.Module):
    def __init__(self,
                 channels,
                 num_heads,
                 proj_drop=0.0,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv="same",
                 padding_q="same",
                 attention_bias=True
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop

        self.conv_q = nn.Conv2d(channels, channels, kernel_size, stride_q, padding_q, bias=attention_bias,
                                groups=channels)
        self.layernorm_q = nn.LayerNorm(channels, eps=1e-5)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,
                                groups=channels)
        self.layernorm_k = nn.LayerNorm(channels, eps=1e-5)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,
                                groups=channels)
        self.layernorm_v = nn.LayerNorm(channels, eps=1e-5)

        self.attention = nn.MultiheadAttention(embed_dim=channels,
                                               bias=attention_bias,
                                               batch_first=True,
                                               # dropout = 0.0,
                                               num_heads=self.num_heads)  # num_heads=self.num_heads)

    def _build_projection(self, x, qkv):

        if qkv == "q":
            x1 = F.relu(self.conv_q(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "k":
            x1 = F.relu(self.conv_k(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "v":
            x1 = F.relu(self.conv_v(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v(x1)
            proj = x1.permute(0, 3, 1, 2)

        return proj

    def forward_conv(self, x1, x2):
        q = self._build_projection(x1, "q")
        k = self._build_projection(x2, "k")
        v = self._build_projection(x2, "v")

        return q, k, v

    def forward(self, x, x2):
        q, k, v = self.forward_conv(x, x2)
        q = q.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        k = k.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        v = v.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)

        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))
        x1 = F.dropout(x1, self.proj_drop)

        return x1


    
class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_q = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_k = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_v = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):  # g : dec  x : enc
        
        q = self.W_q(x)
        k = self.W_k(g)
        v = self.W_v(g)
        psi = self.relu(q + k)
        psi = self.psi(psi)
        out = v * psi
        return q + out    
    


    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) 
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) 

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

    
class Block_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        # self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.upsample = nn.Upsample(scale_factor=2)
        self.layernorm = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        
        self.attn = Attention_block(in_channels, in_channels, in_channels)
        self.mlp = Mlp(in_channels, in_channels * 4, out_channels)
        
        self.convd1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same", dilation=1, groups=out_channels)  #groups=out_channels
        self.convd2 = nn.Conv2d(out_channels, out_channels, 5, 1, padding="same", dilation=1, groups=out_channels)
        self.convd3 = nn.Conv2d(out_channels, out_channels, 7, 1, padding="same", dilation=1, groups=out_channels)
        
        self.conv4 = Conv2d_BN(out_channels * 3, out_channels)

        
        
    def forward(self, x1, skip):   # x1 -> dec    skip -> enc
        
        x1 = self.layernorm(x1)
        x1 = self.upsample(x1)
        x1 = F.relu(self.conv1(x1))
        
        
        x1 = self.attn(x1, skip)
        x1 = self.mlp(x1)
        # x1 = torch.cat((skip, x1), axis=1)
 
        x2 = F.relu(self.convd1(x1))
        x3 = F.relu(self.convd2(x1))
        x4 = F.relu(self.convd3(x1))

        x5 = torch.cat((x2, x3, x4), axis=1)
        x5 = self.conv4(x5)

        
        out = x1 + x5
        return out
    


class Block_encoder_bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        # self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.layernorm = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        
        self.mlp = Mlp(in_channels, in_channels * 4, out_channels)
        self.attn = Attention_block(in_channels, in_channels, in_channels)
        
        self.convd1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same", dilation=1, groups=out_channels)
        self.convd2 = nn.Conv2d(out_channels, out_channels, 5, 1, padding="same", dilation=1, groups=out_channels)
        self.convd3 = nn.Conv2d(out_channels, out_channels, 7, 1, padding="same", dilation=1, groups=out_channels)
                                
        self.conv4 = Conv2d_BN(out_channels * 3, out_channels)
        
        
    def forward(self, x1, skip):   # x -> dec    skip -> enc

        x1 = self.layernorm(x1)
        x1 = F.relu(self.conv1(x1))
        

        x1 = self.attn(x1, skip)
        x1 = self.mlp(x1)
        
        x2 = F.relu(self.convd1(x1))
        x3 = F.relu(self.convd2(x1))
        x4 = F.relu(self.convd3(x1))

        x5 = torch.cat((x2, x3, x4), axis=1)
        x5 = self.conv4(x5)

        out = x1 + x5
        return out    
    
    
    
# class Block_encoder_bottleneck(nn.Module):
#     def __init__(self, blk, in_channels, out_channels, att_heads, dpr):
#         super().__init__()
#         self.blk = blk
#         if ((self.blk == "first") or (self.blk == "bottleneck")):
#             self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
#             self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
#             self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
#             self.trans = Transformer(out_channels, att_heads, dpr)
#             # self.mlp = Mlp(in_channels, in_channels * 4, out_channels)
#             # self.convd1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same", dilation=1, groups=out_channels)
#             # self.convd2 = nn.Conv2d(out_channels, out_channels, 5, 1, padding="same", dilation=1, groups=out_channels)
#             # self.convd3 = nn.Conv2d(out_channels, out_channels, 7, 1, padding="same", dilation=1, groups=out_channels)
#             # self.conv4 = nn.Conv2d(out_channels * 3, out_channels, 1, 1, padding="same")
#             # self.bn4 = nn.BatchNorm2d(out_channels)

#     def forward(self, x, scale_img="none"):

#         x1 = x.permute(0, 2, 3, 1)
#         x1 = self.layernorm(x1)
#         x1 = x1.permute(0, 3, 1, 2)
#         x1 = F.relu(self.conv1(x1))
#         x1 = F.relu(self.conv2(x1))
#         x1 = F.dropout(x1, 0.3)
#         # x1 = F.max_pool2d(x1, (2, 2))
#         out = self.trans(x1, x1)

#         return out

    

class DS_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, padding="same"),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, padding="same"),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, padding="same"),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        out = self.conv3(x1)

        return out    
    
    
