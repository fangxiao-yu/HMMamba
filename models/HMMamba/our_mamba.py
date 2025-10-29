import time
import math
from functools import partial
from typing import Optional, Callable
from .MSFRAM import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import init
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from collections import OrderedDict

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class SKMAttention(nn.Module):

    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        self.vss = VSSBlock(hidden_dim=channel)
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.LeakyReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### 2024.6.28 VSSBLOCK
        vss = self.vss(U.permute(0, 2, 3, 1).contiguous())

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V + vss.permute(0, 3, 1, 2).contiguous()


class MSAA_SKM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAA_SKM, self).__init__()
        self.sk_attention = SKMAttention(in_channels, reduction=8)
        self.out = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=1)

    def forward(self, x1, x2, x4, last=False):
        x_fused = torch.cat([x1, x2, x4], dim=1)
        x_sk = self.sk_attention(x_fused)

        return self.out(x_sk)


class MSAA_MCSM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAA_MCSM, self).__init__()
        # self.Linear = nn.Linear(in_channels,out_channels,bias=False)
        self.mcsm = MCSM(in_channels, out_channels)

    def forward(self, x1, x2, x4, last=False):
        x_fused = torch.cat([x1, x2, x4], dim=1)
        x_mcsm = self.mcsm(x_fused)

        return x_mcsm


class CON_EACAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CON_EACAM, self).__init__()
        self.Linear = nn.Linear(in_channels, out_channels, bias=False)
        # self.eacam1 = EMCAM(in_channels,out_channels)
        self.eacam = EMCAM(in_channels, in_channels)

    def forward(self, x, last=False):
        x_eacam = self.Linear(self.eacam(x).permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        return x_eacam


class CON_MCSM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CON_MCSM, self).__init__()
        self.msfram = MSFRAM(in_channels, out_channels)

    def forward(self, x, last=False):
        msfram = self.msfram(x)

        return msfram


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)
        self.down_2 = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

    def forward(self, x1, x2, x4):
        x_fused = torch.cat([x1, x2, x4], dim=1)
        x_fused = self.down(x_fused)
        x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        x_out = self.up(x_fused_s + x_fused_c)

        return x_out


# class MSAA(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(MSAA, self).__init__()
#         self.fusion_conv = FusionConv(in_channels, out_channels)
#
#     def forward(self, x1, x2, x4, last=False):
#         # # x2 是从低到高，x4是从高到低的设计，x2传递语义信息，x4传递边缘问题特征补充
#         # x_1_2_fusion = self.fusion_1x2(x1, x2)
#         # x_1_4_fusion = self.fusion_1x4(x1, x4)
#         # x_fused = x_1_2_fusion + x_1_4_fusion
#         x_fused = self.fusion_conv(x1, x2, x4)
#         return x_fused


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, padding_mode='reflect',
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channel != out_channel:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.identity = nn.Identity()

    def forward(self, x):
        temp = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.identity(temp)
        out = self.relu(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate, stride=1):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=drop_rate),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=drop_rate),
            nn.LeakyReLU()
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding_mode='reflect', bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(p=drop_rate),
                nn.LeakyReLU(),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        return out


class UpSample1(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate):
        super(UpSample1, self).__init__()
        self.resblock = ResBlock(in_channels=in_channels, out_channels=out_channels * 2, drop_rate=drop_rate)
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channels * 2, out_channels=out_channels, stride=2,
                               kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(drop_rate),
            nn.LeakyReLU()
        )

    def forward(self, x):
        out = self.layer(self.resblock(x))
        return out


class UpSample2(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate):
        super(UpSample2, self).__init__()
        self.resblock1 = ResBlock(in_channels=in_channels, out_channels=out_channels * 2, drop_rate=drop_rate)
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channels * 2, out_channels=out_channels, stride=2,
                               kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(drop_rate),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        out = self.layer(self.resblock1(x))
        return out


class UpSample3(nn.Module):
    def __init__(self, in_channels, num_class, drop_rate):
        super(UpSample3, self).__init__()
        self.resblock = ResBlock(in_channels=in_channels, out_channels=in_channels // 2, drop_rate=drop_rate)
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, stride=2,
                               kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Dropout2d(drop_rate),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=num_class, kernel_size=3, stride=1, padding=1,
                      padding_mode='reflect', bias=False)
        )

    def forward(self, x, feature_map):
        out = self.layer(torch.cat((x, self.resblock(feature_map)), dim=1))
        return out


class Final_classes(nn.Module):
    def __init__(self, in_channels, out_channels, num_class):
        super(Final_classes, self).__init__()
        self.resblock = ResBlock(in_channels=in_channels, out_channels=out_channels, drop_rate=0)
        self.resblock2 = ResBlock(in_channels=out_channels * 2, out_channels=out_channels, drop_rate=0)
        self.conv = nn.Conv2d(in_channels=out_channels, out_channels=num_class, kernel_size=3, stride=1, padding=1,
                              padding_mode='reflect', bias=False)

    def forward(self, x, feature_map):
        out = torch.cat((self.resblock(x), feature_map), dim=1)
        out = self.conv(self.resblock2(out))
        return out


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# class PatchExpand2D(nn.Module):
#     def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim * 2
#         self.dim_scale = dim_scale
#         self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
#         self.norm = norm_layer(self.dim // dim_scale)
#
#     def forward(self, x):
#         B, H, W, C = x.shape
#         x = self.expand(x)
#
#         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
#                       c=C // self.dim_scale)
#         x = self.norm(x)
#
#         return x
class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        # Assuming dim_scale is 2, which means increasing spatial dimensions by a factor of 2
        if dim_scale == 4:
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
            # Adjusting the number of channels after pixel shuffle
        else:
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
            self.adjust_channels = nn.Conv2d(dim // 4, dim // dim_scale, kernel_size=1, stride=1, padding=0, bias=False)

        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        # Pixel shuffle expects the input in the format (B, C, H, W)
        x = rearrange(x, 'b h w c -> b c h w')
        if self.dim_scale == 4:
            x = self.pixel_shuffle(x)
        else:
            x = self.pixel_shuffle(x)
            x = self.adjust_channels(x)
        # Convert back to the original format for normalization
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        return x * y.expand_as(x)


def build_act_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()


class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 ffn_dimsratio=2,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        feedforward_channels = embed_dims * ffn_dimsratio
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        # x = self.fc1(x)
        # x = self.dwconv(x)
        # x = self.act(x)
        # x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        # x = self.fc2(x)
        # x = self.drop(x)
        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            coor=False,
            eca=False,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.coor = coor
        self.ECA = eca
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.weight_gate = nn.Sigmoid()
        # self.coordatt = CoordAtt(inp=self.d_inner,oup=self.d_inner)
        # self.eca = ECAAttention(kernel_size=3)
        # self.emcam = EMCAM(in_channels=self.d_inner,out_channels=self.d_inner)
        self.caffn = ChannelAggregationFFN(embed_dims=self.d_inner, act_type='SiLU', ffn_dimsratio=1)

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)

        assert y1.dtype == torch.float32
        # y1 = torch.transpose(y1, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        # y2 = torch.transpose(y2, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        # y3 = torch.transpose(y3, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        # y4 = torch.transpose(y4, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        # 融合全局和局部特征
        # gate1 = self.weight_gate(y1)
        # gate2 = self.weight_gate(y2)
        # gate3 = self.weight_gate(y3)
        # gate4 = self.weight_gate(y4)
        # x=x.permute(0, 2, 3, 1).contiguous()
        # y1 = gate1 * y1 + (1 - gate1) * x
        # y2 = gate2 * y2 + (1 - gate2) * x
        # y3 = gate3 * y3 + (1 - gate3) * x
        # y4 = gate4 * y4 + (1 - gate4) * x
        # y = y1 + y2 + y3 + y4

        y = self.out_norm(y)
        y = self.caffn(y.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class HM2D(nn.Module):
    def __init__(
            self,
            d_model,
            up_model,
            down_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.up_model = up_model
        self.down_model = down_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.down_sample = nn.Conv2d(in_channels=self.d_inner, out_channels=self.d_inner,
                                     groups=self.d_inner,
                                     bias=conv_bias,
                                     kernel_size=d_conv,
                                     stride=2,
                                     padding=(d_conv - 1) // 2,
                                     **factory_kwargs, )
        self.caffn = ChannelAggregationFFN(embed_dims=self.d_inner, act_type='SiLU', ffn_dimsratio=1)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_up = nn.Linear(self.up_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_down = nn.Linear(self.down_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )

        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)

        del self.x_proj

        self.x_proj_u = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight_u = nn.Parameter(torch.stack([t.weight for t in self.x_proj_u], dim=0))  # (K=1, N, inner)

        del self.x_proj_u

        self.x_proj_d = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight_d = nn.Parameter(torch.stack([t.weight for t in self.x_proj_d], dim=0))  # (K=1, N, inner)

        del self.x_proj_d

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.dt_projs_u = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),

        )
        self.dt_projs_weight_u = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs_u], dim=0))  # (K=1, inner, rank)
        self.dt_projs_bias_u = nn.Parameter(torch.stack([t.bias for t in self.dt_projs_u], dim=0))
        del self.dt_projs_u

        self.dt_projs_d = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),

        )
        self.dt_projs_weight_d = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs_d], dim=0))  # (K=1, inner, rank)

        # self.selective_scan = selective_scan_fn
        self.dt_projs_bias_d = nn.Parameter(torch.stack([t.bias for t in self.dt_projs_d], dim=0))
        del self.dt_projs_d

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.A_logs_u = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=1, D, N)
        self.Ds_u = self.D_init(self.d_inner, copies=1, merge=True)  # (K=1, D, N)

        self.A_logs_d = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=2, D, N)
        self.Ds_d = self.D_init(self.d_inner, copies=2, merge=True)  # (K=2, D, N)

        self.forward_core = self.forward_corev0
        # self.forward_core_u = self.forward_corev_up

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward_corev_up_down(self, x: torch.Tensor, up=True):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 1
        if up:
            # xs = torch.flip(x.view(B, -1, L), dims=[-1])
            xs = torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        else:
            # xs = x.view(B, -1, L)
            xs = torch.flip(torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),dims=[-1])

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight_u)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight_u)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds_u.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs_u.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias_u.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        if up:
            inv_y = torch.transpose(out_y[:, 0].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            # inv_y = torch.flip(out_y[:, 0], dims=[-1])

        else:
            inv_y = torch.flip(torch.transpose(out_y[:, 0].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L),
                         dims=[-1])

        return inv_y

    def forward_corev(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 2

        xs = torch.stack([torch.flip(x.view(B, -1, L), dims=[-1]),
                          x.view(B, -1, L)],
                         dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight_d)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight_d)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds_d.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs_d.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias_d.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # y1 = torch.transpose(out_y[:, 0].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y1 = torch.flip(out_y[:, 0], dims=[-1])
        # y2 = torch.flip(torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L),
        #                 dims=[-1])

        return y1, out_y[:,1]

    def forward(self, x: torch.Tensor, up: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        up_xz = self.in_proj_up(up)
        # down_xz = self.in_proj_down(down)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        up_x, up_z = up_xz.chunk(2, dim=-1)

        # down_x, down_z = down_xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        up_x = up_x.permute(0, 3, 1, 2).contiguous()
        # down_x = down_x.permute(0, 3, 1, 2).contiguous()
        down_x = self.act(self.down_sample(x))
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        up_x = self.act(self.conv2d(up_x))

        # down_x = self.act(self.conv2d(down_x))
        # x_up = F.interpolate(x, scale_factor=2)
        y_up = self.forward_corev_up_down(up_x)
        y_up = torch.transpose(y_up, dim0=1, dim1=2).contiguous().view(B, 2 * H, 2 * W, -1)
        y1,y2 = self.forward_corev(x)
        y1 = torch.transpose(y1, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y2 = torch.transpose(y2, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        # x_down = self.down_sample(x)
        y_down= self.forward_corev_up_down(down_x,up=False)

        y_down = torch.transpose(y_down, dim0=1, dim1=2).contiguous().view(B, int(H / 2), int(W / 2), -1)
        # y_down2 = torch.transpose(y_down2, dim0=1, dim1=2).contiguous().view(B, int(H / 2), int(W / 2), -1)
        # y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y_up = self.down_sample(y_up.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        up_z = self.down_sample(up_z.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        y_down = F.interpolate(y_down.permute(0, 3, 1, 2).contiguous(), scale_factor=2,mode='nearest').permute(0, 2, 3,
                                                                                                  1).contiguous()
        # y_down2 = F.interpolate(y_down2.permute(0, 3, 1, 2).contiguous(), scale_factor=2).permute(0, 2, 3,
        #                                                                                           1).contiguous()
        # down_z = F.interpolate(down_z.permute(0, 3, 1, 2).contiguous(), scale_factor=2).permute(0, 2, 3,
        #                                                                                         1).contiguous()
        y = y_up + y1+y2 + y_down
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = self.caffn(y.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        y = y * F.silu(z) * F.silu(up_z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            coor=False,
            eca=False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, coor=coor, eca=eca,
                                   **kwargs)
        self.drop_path = DropPath(drop_path)

        # self.coor = CoordAtt(inp=hidden_dim,oup=hidden_dim)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        # x_ca = self.coor(x.permute(0, 3, 1, 2).contiguous())
        # x = x + x_ca.permute(0, 2, 3, 1).contiguous()
        # print("resm3")
        return x


class HMMBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            up_memory_dim=0,
            down_memory_dim=0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            coor=False,
            eca=False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_up = norm_layer(up_memory_dim)
        self.ln_down = norm_layer(down_memory_dim)
        self.linear_up = nn.Linear(up_memory_dim, hidden_dim)
        self.linear_down = nn.Linear(down_memory_dim, hidden_dim)
        self.dwconv = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                                groups=hidden_dim,
                                bias=True,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                )
        self.self_attention = HM2D(d_model=hidden_dim, up_model=up_memory_dim, down_model=down_memory_dim,
                                   dropout=attn_drop_rate, d_state=d_state,
                                   **kwargs)
        self.drop_path = DropPath(drop_path)
        self.act = nn.SiLU()

    def forward(self, input: torch.Tensor, up: torch.Tensor, down: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input), self.ln_up(up)))
        #可以加激活函数
        up = self.dwconv(self.linear_up(up).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        # down = self.linear_down(
        #     F.interpolate(down.permute(0, 3, 1, 2).contiguous(), scale_factor=2).permute(0, 2, 3, 1).contiguous())

        return x + up
