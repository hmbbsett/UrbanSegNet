import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import LocalDINOv3Encoder
from .mamba_core import SS2D


class Conv2d_BN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.add_module('conv',
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_channels))


class MKDWConv(nn.Module):

    def __init__(self, dim, n_groups=3):
        super().__init__()
        assert dim % n_groups == 0, f"Local dim {dim} must be divisible by n_groups {n_groups}"
        self.split_dim = dim // n_groups
        self.n_groups = n_groups

        self.convs = nn.ModuleList()
        for i in range(n_groups):
            k = 3 + 2 * i
            p = k // 2
            self.convs.append(
                nn.Conv2d(self.split_dim, self.split_dim, kernel_size=k, stride=1, padding=p, groups=self.split_dim,
                          bias=False)
            )

        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()

    def forward(self, x):
        splits = torch.split(x, self.split_dim, dim=1)

        outputs = []
        for i, conv in enumerate(self.convs):
            outputs.append(conv(splits[i]))

        out = torch.cat(outputs, dim=1)

        out = self.bn(out)
        out = self.act(out)
        return out


class MobileMambaDecoderBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, ssm_ratio=2.0, g_ratio=0.5, l_ratio=0.25):
        super().__init__()

        def _make_divisible(v, divisor=8):
            return int(round(v / divisor) * divisor)

        self.g_dim = _make_divisible(dim * g_ratio, 16)
        self.l_dim = _make_divisible(dim * l_ratio, 24)
        self.i_dim = dim - self.g_dim - self.l_dim

        self.norm_mamba = nn.LayerNorm(self.g_dim)
        self.mamba = SS2D(d_model=self.g_dim, d_state=16, ssm_ratio=ssm_ratio, channel_first=True)

        self.local_conv = MKDWConv(self.l_dim, n_groups=3)

        self.mrffi_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.mrffi_norm = nn.BatchNorm2d(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.norm_ffn = nn.BatchNorm2d(dim)
        self.ffn = nn.Sequential(
            Conv2d_BN(dim, hidden_dim, 1),
            nn.SiLU(),
            Conv2d_BN(hidden_dim, dim, 1)
        )

    def forward(self, x):
        shortcut = x

        x_g, x_l, x_id = torch.split(x, [self.g_dim, self.l_dim, self.i_dim], dim=1)

        x_g_in = x_g.permute(0, 2, 3, 1)
        x_g_in = self.norm_mamba(x_g_in)
        x_g_in = x_g_in.permute(0, 3, 1, 2)
        x_g_out = self.mamba(x_g_in)

        x_l_out = self.local_conv(x_l)

        x_id_out = x_id

        x_mixed = torch.cat([x_g_out, x_l_out, x_id_out], dim=1)
        x_mixed = self.mrffi_proj(x_mixed)
        x_mixed = self.mrffi_norm(x_mixed)

        x = shortcut + x_mixed

        x = x + self.ffn(self.norm_ffn(x))

        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UrbanSeg(nn.Module):
    def __init__(self, num_classes=9, img_size=1024, r=128):
        super().__init__()

        ckpt_path = "weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

        self.encoder = LocalDINOv3Encoder(
            ckpt_path=ckpt_path,
            img_size=img_size,
            use_lora=True,
            lora_rank=r
        )

        self.embed_dim = self.encoder.embed_dim

        self.decoder_in = Conv2d_BN(self.embed_dim, 512, 3, 1, 1)

        self.mamba_layers = nn.ModuleList([
            MobileMambaDecoderBlock(512),
            MobileMambaDecoderBlock(512),
            MobileMambaDecoderBlock(512)
        ])

        self.up_16_to_8 = UpsampleBlock(512, 64)
        self.up_8_to_4 = UpsampleBlock(64, 32)

        self.head = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(32, num_classes, 1)
        )

        self.aux_head = nn.Conv2d(self.embed_dim, num_classes, 1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        x_patch = self.encoder(x)
        x_dec = self.decoder_in(x_patch)

        for layer in self.mamba_layers:
            x_dec = layer(x_dec)

        x_8 = self.up_16_to_8(x_dec)
        x_4 = self.up_8_to_4(x_8)  # [B, 32, H/4, W/4]

        logits = self.head(x_4)

        logits_aux = self.aux_head(x_patch)

        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        logits_aux = F.interpolate(logits_aux, size=(H, W), mode='bilinear', align_corners=False)

        return logits_aux, logits