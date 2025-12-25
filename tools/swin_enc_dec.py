# tools/swin_enc_dec.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.swin_transformer import SwinTransformer
from tools.filters import RampFilter, HannFilter


# -------------------------
# Basic blocks
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


# -------------------------
# Swin Stage-1 Model
# -------------------------
class SwinUNetStage1(nn.Module):
    """
    Swin Transformer encoder + convolutional decoder
    Input : [B, 1, H, W]
    Output: [B, 1, H, W]
    """

    def __init__(
        self,
        filter_type="ramp",
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        pretrained=True,
    ):
        super().__init__()

        # ---- Sinogram frequency filter ----
        if filter_type == "ramp":
            self.sino_filter = RampFilter()
        elif filter_type == "hann":
            self.sino_filter = HannFilter()
        else:
            self.sino_filter = nn.Identity()

        # ---- Channel lift ----
        self.input_proj = nn.Conv2d(1, 3, kernel_size=1)

        # ---- Swin encoder ----
        self.encoder = SwinTransformer(
            img_size=384,
            patch_size=4,
            in_chans=3,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            pretrained=pretrained,
        )

        # ---- Decoder ----
        self.up1 = UpBlock(embed_dim * 8, embed_dim * 4)  # 12 → 24
        self.up2 = UpBlock(embed_dim * 4, embed_dim * 2)  # 24 → 48
        self.up3 = UpBlock(embed_dim * 2, embed_dim)      # 48 → 96
        self.up4 = UpBlock(embed_dim, embed_dim)          # 96 → 192

        self.out_conv = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, x):
        # Original spatial size (362×362)
        h, w = x.shape[2], x.shape[3]

        # ---- Pad to 384×384 ----
        x = F.pad(x, (0, 384 - w, 0, 384 - h))

        # ---- Sinogram filter ----
        x = self.sino_filter(x)

        # ---- Channel lift ----
        x = self.input_proj(x)

        # ---- Swin encoder ----
        feat = self.encoder.forward_features(x)

        # ---- Normalize Swin output to [B, C, H, W] ----
        if feat.dim() == 3:
            B, L, C = feat.shape
            S = int(L ** 0.5)
            feat = feat.transpose(1, 2).reshape(B, C, S, S)
        elif feat.dim() == 4:
            if feat.shape[1] != feat.shape[-1]:
                feat = feat.permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError(f"Unexpected Swin output shape: {feat.shape}")

        # ---- Decoder ----
        x = self.up1(feat)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        # ---- Upsample to padded resolution (384×384) ----
        x = F.interpolate(x, size=(384, 384), mode="bilinear", align_corners=False)

        out = self.out_conv(x)

        # ---- Crop back to original size (362×362) ----
        return out[:, :, :h, :w]
