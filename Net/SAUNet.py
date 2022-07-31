from typing import Tuple, Union
import torch.nn as nn
from Net.ViT import *
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock


class SAUNet(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: Tuple[int, int, int],
            feature_size: int = 16,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = False,
            res_block: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.pe = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads
        )
        self.ln1 = nn.LayerNorm(hidden_size)

        self.ViT_Modules = nn.ModuleList(
            [
                ViT_Module(
                    in_channels=in_channels,
                    img_size=img_size,
                    patch_size=self.patch_size,
                    hidden_size=hidden_size,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads
                )
                for i in range(self.num_layers)
            ]
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore

        # use with HM
        # self.outblock1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size,
        #                               out_channels=out_channels)  # type: ignore
        # self.outblock2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2,
        #                               out_channels=out_channels)  # type: ignore
        # self.outblock3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4,
        #                               out_channels=out_channels)  # type: ignore
        # self.outblock4 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 8,
        #                               out_channels=out_channels)  # type: ignore
        # self.outblock5 = UnetOutBlock(spatial_dims=3, in_channels=hidden_size,
        #                               out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        enc1 = self.encoder1(x_in)

        x = self.pe(x_in)
        layer_out = []
        for ViT_Module in self.ViT_Modules:
            x = ViT_Module(x)
            layer_out.append(x)
        x = self.ln1(x)

        x2 = layer_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = layer_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = layer_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))  # feat_size * 8
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)  # feat_size
        dec3 = self.decoder5(dec4, enc4)  # feat_size * 8
        dec2 = self.decoder4(dec3, enc3)  # feat_size * 4
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)

        # use with HM
        # out1 = self.outblock1(out)
        # out2 = self.outblock2(dec1)
        # out3 = self.outblock3(dec2)
        # out4 = self.outblock4(dec3)
        # out5 = self.outblock5(dec4)

        logits = self.out(out)
        # use with HM
        # return out1, out2, out3, out4, out5

        return logits
