import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

from tools.filters import RampFilter, HannFilter


class UpBlock(nn.Module):
    """
    Decoder block with skip connection:
    - up: ConvTranspose2d(in_ch -> out_ch)
    - concat with skip (skip_ch), then 3x convs to refine -> out_ch
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.bn_up = nn.BatchNorm2d(out_ch)
        self.conv1 = nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.relu(self.bn_up(self.up(x)))
        # ensure skip and upsampled feature align in size
        if x.shape[2:] != skip.shape[2:]:
            min_h = min(x.shape[2], skip.shape[2])
            min_w = min(x.shape[3], skip.shape[3])
            x = x[:, :, :min_h, :min_w]
            skip = skip[:, :, :min_h, :min_w]
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class ResNet34UNet(nn.Module):
    """
    ResNet34 encoder + UNet decoder with real skip connections.
    Input/Output: [B,1,H,W] where H,W can be arbitrary.
    
    - Automatically pads input to nearest multiple of 32
    - Crops output back to original H×W
    - Optional sinogram-domain frequency filter
    """
    def __init__(self, pretrained: bool = False, filter_type: str = "ramp"):
        super().__init__()

        # ---- Sinogram filter (no learnable params) ----
        ft = (filter_type or "none").lower()
        if ft == "ramp":
            self.sino_filter = RampFilter()
        elif ft == "hann":
            self.sino_filter = HannFilter()
        else:
            self.sino_filter = nn.Identity()

        # ---- ResNet encoder ----
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = resnet34(weights=weights)

        # stem for single-channel input
        self.stem_conv = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.stem_conv.weight.copy_(resnet.conv1.weight.sum(dim=1, keepdim=True))
        self.stem_bn   = resnet.bn1
        self.stem_relu = resnet.relu
        self.stem_pool = resnet.maxpool

        # encoder
        self.enc1 = resnet.layer1   # 64 ch
        self.enc2 = resnet.layer2   # 128 ch
        self.enc3 = resnet.layer3   # 256 ch
        self.enc4 = resnet.layer4   # 512 ch

        # decoder with skips
        self.dec4 = UpBlock(512, 256, 256)
        self.dec3 = UpBlock(256, 128, 128)
        self.dec2 = UpBlock(128, 64,  64)
        self.dec1 = UpBlock(64,  64,  64)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Save original size
        h, w = x.shape[2], x.shape[3]

        # Pad to nearest multiple of 32 (UNet-friendly)
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        x = F.pad(x, (0, pad_w, 0, pad_h))  # pad (left,right,top,bottom)

        # ---- Frequency filter ----
        x = self.sino_filter(x)

        # ---- Encoder ----
        x0 = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        x1 = self.enc1(self.stem_pool(x0))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # ---- Decoder ----
        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        out = self.out_conv(d1)

        # Crop back to original H×W
        out = out[:, :, :h, :w]
        return out


class RefinerUNet(nn.Module):
    """
    Simple UNet-like model for image-to-image refinement.
    Preserves input dimensions exactly.
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn_bottleneck = nn.BatchNorm2d(512)
        
        # Decoder
        self.dec3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.bn_dec3 = nn.BatchNorm2d(256)
        self.conv_dec3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # 512 because of skip connection
        self.bn_conv_dec3 = nn.BatchNorm2d(256)
        
        self.dec2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.bn_dec2 = nn.BatchNorm2d(128)
        self.conv_dec2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 256 because of skip connection
        self.bn_conv_dec2 = nn.BatchNorm2d(128)
        
        self.dec1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_dec1 = nn.BatchNorm2d(64)
        self.conv_dec1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 128 because of skip connection
        self.bn_conv_dec1 = nn.BatchNorm2d(64)
        
        # Output
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # Store original size
        original_h, original_w = x.shape[2], x.shape[3]
        
        # Encoder
        e1 = self.relu(self.bn1(self.enc1(x)))
        e2 = self.relu(self.bn2(self.enc2(e1)))
        e3 = self.relu(self.bn3(self.enc3(e2)))
        
        # Bottleneck
        bottleneck = self.relu(self.bn_bottleneck(self.bottleneck(e3)))
        
        # Decoder with skip connections
        d3 = self.relu(self.bn_dec3(self.dec3(bottleneck)))
        # Handle potential size mismatch for skip connection
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.relu(self.bn_conv_dec3(self.conv_dec3(d3)))
        
        d2 = self.relu(self.bn_dec2(self.dec2(d3)))
        # Handle potential size mismatch for skip connection
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.relu(self.bn_conv_dec2(self.conv_dec2(d2)))
        
        d1 = self.relu(self.bn_dec1(self.dec1(d2)))
        # Handle potential size mismatch for skip connection
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.relu(self.bn_conv_dec1(self.conv_dec1(d1)))
        
        # Output
        out = self.out_conv(d1)
        
        # Ensure output matches input size exactly
        if out.shape[2:] != (original_h, original_w):
            out = F.interpolate(out, size=(original_h, original_w), mode='bilinear', align_corners=False)
        
        return out
        