"""CNN/Vision architectures: CNN, ResNet, U-Net, EfficientNet, MobileNet, ViT, SAM.

Papers: Krizhevsky 2012, He 2015, Ronneberger 2015, Tan 2019, Howard 2017,
Dosovitskiy 2020, Kirillov 2023.
"""

from __future__ import annotations

import math
import random

from .registry import register


class Conv2d:
    """2D convolution layer."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int = 0
    ) -> None:
        s = math.sqrt(2.0 / (in_ch * kernel * kernel))
        self.w = [
            [
                [[random.gauss(0, s) for _ in range(kernel)] for _ in range(kernel)]
                for _ in range(in_ch)
            ]
            for _ in range(out_ch)
        ]
        self.b = [0.0] * out_ch
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def forward(self, x: list[list[list[float]]]) -> list[list[list[float]]]:
        h = len(x)
        w = len(x[0]) if h > 0 else 0
        oh = (h + 2 * self.padding - self.kernel) // self.stride + 1
        ow = (w + 2 * self.padding - self.kernel) // self.stride + 1
        padded = [
            [[0.0] * (w + 2 * self.padding) for _ in range(h + 2 * self.padding)]
            for _ in range(len(x))
        ]
        for c in range(len(x)):
            for i in range(h):
                for j in range(w):
                    padded[c][i + self.padding][j + self.padding] = x[c][i][j]
        out = [[[0.0] * ow for _ in range(oh)] for _ in range(len(self.w))]
        for o in range(len(self.w)):
            for i in range(oh):
                for j in range(ow):
                    s = self.b[o]
                    for ci in range(len(self.w[o])):
                        for ki in range(self.kernel):
                            for kj in range(self.kernel):
                                s += (
                                    self.w[o][ci][ki][kj]
                                    * padded[ci][i * self.stride + ki][j * self.stride + kj]
                                )
                    out[o][i][j] = max(0.0, s)
        return out


register("cnn.conv2d", Conv2d)


class ResNetBlock:
    """ResNet residual block (He et al. 2015)."""

    def __init__(self, channels: int) -> None:
        self.conv1 = Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = Conv2d(channels, channels, 3, 1, 1)
        self.channels = channels

    def forward(self, x: list[list[list[float]]]) -> list[list[list[float]]]:
        shortcut = x
        out = self.conv1.forward(x)
        out = self.conv2.forward(out)
        for c in range(min(len(out), len(shortcut))):
            for i in range(min(len(out[0]), len(shortcut[0]))):
                for j in range(min(len(out[0][0]), len(shortcut[0][0]))):
                    out[c][i][j] += shortcut[c][i][j]
                    out[c][i][j] = max(0.0, out[c][i][j])
        return out


register("cnn.resnet_block", ResNetBlock)


class UNet:
    """U-Net (Ronneberger, Fischer, Brox 2015)."""

    def __init__(self, in_ch: int, base_ch: int = 64) -> None:
        self.enc1 = Conv2d(in_ch, base_ch, 3, 1, 1)
        self.enc2 = Conv2d(base_ch, base_ch * 2, 3, 1, 1)
        self.enc3 = Conv2d(base_ch * 2, base_ch * 4, 3, 1, 1)
        self.dec3 = Conv2d(base_ch * 4 + base_ch * 2, base_ch * 2, 3, 1, 1)
        self.dec2 = Conv2d(base_ch * 2 + base_ch, base_ch, 3, 1, 1)
        self.dec1 = Conv2d(base_ch, in_ch, 3, 1, 1)

    def forward(self, x: list[list[list[float]]]) -> list[list[list[float]]]:
        e1 = self.enc1.forward(x)
        e2 = self.enc2.forward([[row[::2] for row in e1c[::2]] for e1c in e1])
        e3 = self.enc3.forward([[row[::2] for row in e2c[::2]] for e2c in e2])
        d3 = self.dec3.forward(e3 + [row for row in e2])
        d2 = self.dec2.forward(d3 + [row for row in e1])
        return self.dec1.forward(d2)


register("cnn.unet", UNet)


class PatchEmbed:
    """ViT patch embedding (Dosovitskiy et al. 2020)."""

    def __init__(
        self, img_size: int = 224, patch_size: int = 16, in_ch: int = 3, d_model: int = 768
    ) -> None:
        self.patch_size = patch_size
        n_patches = (img_size // patch_size) ** 2
        s = math.sqrt(2.0 / (in_ch * patch_size * patch_size))
        self.proj = [
            [random.gauss(0, s) for _ in range(in_ch * patch_size * patch_size)]
            for _ in range(d_model)
        ]
        self.cls_token = [random.gauss(0, 0.1) for _ in range(d_model)]
        self.pos_embed = [
            [random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(n_patches + 1)
        ]

    def forward(self, x: list[list[list[list[float]]]]) -> list[list[float]]:
        b = len(x)
        c = len(x[0])
        h = len(x[0][0])
        w = len(x[0][0][0])
        ps = self.patch_size
        tokens = [list(self.cls_token)]
        for i in range(b):
            for j in range(0, h, ps):
                for k in range(0, w, ps):
                    flat = []
                    for ch in range(c):
                        for dj in range(ps):
                            for dk in range(ps):
                                flat.append(
                                    x[i][ch][j + dj][k + dk] if j + dj < h and k + dk < w else 0.0
                                )
                    tok = [
                        sum(self.proj[d][di] * flat[di] for di in range(len(flat)))
                        for d in range(len(self.proj))
                    ]
                    tokens.append(tok)
        return [self.cls_token] + tokens


register("cnn.vit_patch_embed", PatchEmbed)


class VisionTransformer:
    """ViT (Dosovitskiy et al. 2020)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
    ) -> None:
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, d_model)
        from .transformer import TransformerBlock

        self.blocks = [TransformerBlock(d_model, n_heads, d_model * 4) for _ in range(n_layers)]
        self.norm = lambda x: [
            (v - sum(x) / len(x))
            / (math.sqrt(sum((vi - sum(x) / len(x)) ** 2 for vi in x) / len(x)) + 1e-6)
            for v in x
        ]

    def forward(self, x: list[list[list[list[float]]]]) -> list[float]:
        tokens = self.patch_embed.forward(x)
        for block in self.blocks:
            tokens = block.forward(tokens)
        return [tokens[0]] if tokens else [0.0]  # CLS token


register("cnn.vit", VisionTransformer)
