import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Network(nn.Module):
    def __init__(
        self,
        input_shape=(3, 128, 128),
        num_classes=5,
        dim=[32, 64, 128, 256, 512],
        dropout=0.1,
    ):
        super().__init__()
        self.dim = [input_shape[0]] + dim

        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(len(self.dim) - 2):
            self.encoders.append(
                DoubleConv(self.dim[i], self.dim[i + 1], dropout=dropout)
            )
            self.encoders.append(nn.MaxPool2d(2))
        self.encoders.append(DoubleConv(self.dim[-2], self.dim[-1], dropout=dropout))

        # Bottleneck
        self.pos_embed = nn.Conv2d(self.dim[-1], self.dim[-1], kernel_size=1)
        self.bottleneck = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=self.dim[-1],
                    nhead=4,
                    dim_feedforward=self.dim[-1] * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(2)
            ]
        )

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(len(self.dim) - 2):
            self.decoders.append(
                nn.ConvTranspose2d(
                    self.dim[-(i + 1)], self.dim[-(i + 2)], kernel_size=2, stride=2
                )
            )
            self.decoders.append(
                DoubleConv(self.dim[-(i + 1)], self.dim[-(i + 2)], dropout=dropout)
            )

        # Final output
        self.output = nn.Conv2d(self.dim[1], num_classes, kernel_size=1)

    def forward(self, x):
        skip_x = []
        # Encoder
        for i in range(len(self.dim) - 2):
            x = self.encoders[2 * i](x)  # Conv
            skip_x.append(x)
            x = self.encoders[2 * i + 1](x)  # Pool
        x = self.encoders[-1](x)  # Last Conv

        # Transformer bottleneck
        x = x + self.pos_embed(x)
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # B, H*W, C
        x = self.bottleneck(x)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # B, C, H, W

        # Decoder
        for i in range(len(self.dim) - 2):
            x = self.decoders[2 * i](x)  # UpConv
            x = torch.cat([x, skip_x[-(i + 1)]], dim=1)  # Skip connection
            x = self.decoders[2 * i + 1](x)  # Conv

        return self.output(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))

    def get_parameter(self):
        return {}


if __name__ == "__main__":
    input_shape = (3, 256, 256)
    num_classes = 5
    model = Network(input_shape, num_classes)
    x = torch.randn(1, *input_shape)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
