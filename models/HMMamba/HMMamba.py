from .our_mamba import *

import torch
from torch import nn
import os
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks import ResidualUnit


class Resm_mamba3(nn.Module):
    def __init__(self, num_classes, input_channel=3, drop_path_rate=0.2, norm_layer=nn.LayerNorm):
        super(Resm_mamba3, self).__init__()
        self.num_classes = num_classes
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 8)]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, 8)][::-1]
        # self.AFFs = nn.ModuleList([
        #     MSAA_MCSM(224, 48),
        #     MSAA_MCSM(224, 96),
        #     MSAA_MCSM(224, 192),
        # ])
        self.AFFs = nn.ModuleList([
            CON_MCSM(96, 48),
            CON_MCSM(192, 96),
            CON_MCSM(384, 192),
        ])
        self.HMS = nn.ModuleList([
            HMMBlock(hidden_dim=384, up_memory_dim=96, down_memory_dim=768, drop_path=dpr_decoder[2],
                     norm_layer=norm_layer),
            HMMBlock(hidden_dim=384, up_memory_dim=96, down_memory_dim=768, drop_path=dpr_decoder[3],
                     norm_layer=norm_layer),
            HMMBlock(hidden_dim=192, up_memory_dim=48, down_memory_dim=192, drop_path=dpr_decoder[4],
                     norm_layer=norm_layer),
            HMMBlock(hidden_dim=192, up_memory_dim=48, down_memory_dim=192, drop_path=dpr_decoder[5],
                     norm_layer=norm_layer),
            HMMBlock(hidden_dim=96, up_memory_dim=48, down_memory_dim=96, drop_path=dpr_decoder[6],
                     norm_layer=norm_layer),
            HMMBlock(hidden_dim=96, up_memory_dim=48, down_memory_dim=96, drop_path=dpr_decoder[7],
                     norm_layer=norm_layer),
        ])
        # self.transfer = nn.ModuleList(
        #     [
        #         nn.Conv2d(96, 32, 1, bias=False),
        #         nn.Conv2d(192, 64, 1, bias=False),
        #         nn.Conv2d(384, 128, 1, bias=False),
        #     ]
        # )

        self.res_down1 = ResBlock(in_channels=input_channel, out_channels=48, drop_rate=dpr[0],
                                  stride=2)  # b 48 h/2 w/2
        self.res_down2 = ResBlock(in_channels=48, out_channels=96, drop_rate=dpr[2],
                                  stride=2)  # b 96 h/4 w/4
        self.res_down3 = ResBlock(in_channels=96, out_channels=192, drop_rate=dpr[4],
                                  stride=2)  # b 192 h/8 w/8
        self.res_down4 = ResBlock(in_channels=192, out_channels=384, drop_rate=dpr[6],
                                  stride=2)  # b 384 h/16 w/16
        self.res_down5 = ResBlock(in_channels=384, out_channels=768, drop_rate=dpr[7],
                                  stride=2)  # b 768 h/32 w/32

        self.vss_up11 = VSSBlock(hidden_dim=768, drop_path=dpr_decoder[0], norm_layer=norm_layer)
        self.vss_up12 = VSSBlock(hidden_dim=768, drop_path=dpr_decoder[1], norm_layer=norm_layer)
        self.exp1 = PatchExpand2D(dim=768, dim_scale=4)

        self.vss_up21 = VSSBlock(hidden_dim=384, drop_path=dpr_decoder[2], norm_layer=norm_layer)
        self.vss_up22 = VSSBlock(hidden_dim=384, drop_path=dpr_decoder[3], norm_layer=norm_layer)
        self.exp2 = PatchExpand2D(dim=384, dim_scale=4)

        self.vss_up31 = VSSBlock(hidden_dim=192, drop_path=dpr_decoder[4], norm_layer=norm_layer)
        self.vss_up32 = VSSBlock(hidden_dim=192, drop_path=dpr_decoder[5], norm_layer=norm_layer)
        self.exp3 = PatchExpand2D(dim=192, dim_scale=4)

        self.vss_up41 = VSSBlock(hidden_dim=96, drop_path=dpr_decoder[6], norm_layer=norm_layer)
        self.vss_up42 = VSSBlock(hidden_dim=96, drop_path=dpr_decoder[7], norm_layer=norm_layer)
        self.exp4 = PatchExpand2D(dim=96, dim_scale=2)

        self.vss_up51 = VSSBlock(hidden_dim=96, drop_path=dpr_decoder[7], norm_layer=norm_layer)
        self.vss_up52 = VSSBlock(hidden_dim=96, drop_path=dpr_decoder[7], norm_layer=norm_layer)
        self.out = nn.Sequential(

            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(48),
            # nn.Dropout2d(0.2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(48),
            # nn.Dropout2d(0.2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=48, out_channels=num_classes, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        s1 = self.res_down1(x)
        s2 = self.res_down2(s1)
        s3 = self.res_down3(s2)
        s4 = self.res_down4(s3)
        s5 = self.res_down5(s4)  # b 768 h/32 w/32

        s2 = self.AFFs[0](s2)
        s3 = self.AFFs[1](s3)
        s4 = self.AFFs[2](s4)

        up1 = self.exp1(self.vss_up12(self.vss_up11(s5.permute(0, 2, 3, 1).contiguous())))  # b h/16 w/16 192

        # s2_2 = self.transfer[0](s2)
        # s2_3 = F.interpolate(self.transfer[0](s2), scale_factor=0.5, mode='nearest')
        # s2_4 = F.interpolate(self.transfer[0](s2), scale_factor=0.25, mode='nearest')
        #
        # s3_3 = self.transfer[1](s3)
        # s3_2 = F.interpolate(self.transfer[1](s3), scale_factor=2, mode='nearest')
        # s3_4 = F.interpolate(self.transfer[1](s3), scale_factor=0.5, mode='nearest')
        #
        # s4_4 = self.transfer[2](s4)
        # s4_3 = F.interpolate(self.transfer[2](s4), scale_factor=2, mode='nearest')
        # s4_2 = F.interpolate(self.transfer[2](s4), scale_factor=4, mode='nearest')

        # u2 = torch.cat((up1.permute(0, 3, 1, 2).contiguous(), self.AFFs[2](s4_4, s3_4, s2_4)), dim=1)
        # u2 = torch.cat((up1.permute(0, 3, 1, 2).contiguous(), self.AFFs[2](s4)), dim=1)
        # up2 = self.exp2(self.vss_up22(self.vss_up21(u2.permute(0, 2, 3, 1).contiguous())))  # b h/8 w/8 96
        u2 = torch.cat((up1.permute(0, 3, 1, 2).contiguous(), s4), dim=1)
        up2 = self.exp2(
            self.HMS[1](self.HMS[0](u2.permute(0, 2, 3, 1).contiguous(), s3.permute(0, 2, 3, 1).contiguous(),
                                    s5.permute(0, 2, 3, 1).contiguous()), s3.permute(0, 2, 3, 1).contiguous(),
                        s5.permute(0, 2, 3, 1).contiguous()))  # b h/8 w/8 96

        # u3 = torch.cat((up2.permute(0, 3, 1, 2).contiguous(), self.AFFs[1](s3_3, s2_3, s4_3)), dim=1)
        # u3 = torch.cat((up2.permute(0, 3, 1, 2).contiguous(), self.AFFs[1](s3)), dim=1)
        # up3 = self.exp3(self.vss_up32(self.vss_up31(u3.permute(0, 2, 3, 1).contiguous())))  # b h/4 w/4 48
        u3 = torch.cat((up2.permute(0, 3, 1, 2).contiguous(), s3), dim=1)
        up3 = self.exp3(
            self.HMS[3](self.HMS[2](u3.permute(0, 2, 3, 1).contiguous(), s2.permute(0, 2, 3, 1).contiguous(), up1),
                        s2.permute(0, 2, 3, 1).contiguous(), up1))  # b h/4 w/4 48

        # u4 = torch.cat((up3.permute(0, 3, 1, 2).contiguous(), self.AFFs[0](s2_2, s3_2, s4_2)), dim=1)
        # u4 = torch.cat((up3.permute(0, 3, 1, 2).contiguous(), self.AFFs[0](s2)), dim=1)
        # up4 = self.exp4(self.vss_up42(self.vss_up41(u4.permute(0, 2, 3, 1).contiguous())))  # b h/2 w/2 48
        u4 = torch.cat((up3.permute(0, 3, 1, 2).contiguous(), s2), dim=1)
        up4 = self.exp4(self.HMS[5](
            self.HMS[4](u4.permute(0, 2, 3, 1).contiguous(), s1.permute(0, 2, 3, 1).contiguous(),
                        up2), s1.permute(0, 2, 3, 1).contiguous(),
            up2))  # b h/2 w/2 48

        u5 = torch.cat((up4.permute(0, 3, 1, 2).contiguous(), s1), dim=1)
        up5 = self.vss_up52(self.vss_up51(u5.permute(0, 2, 3, 1).contiguous()))
        up5 = self.out(
            F.interpolate(up5.permute(0, 3, 1, 2).contiguous(), scale_factor=2, mode='nearest'))  # b num_class h w

        if self.num_classes == 1:
            return torch.sigmoid(up5)
        else:
            return up5


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    x = torch.randn(2, 3, 224, 224).cuda()
    resmamba = Resm_mamba3(num_classes=9).cuda()
    print(resmamba(x).shape)
