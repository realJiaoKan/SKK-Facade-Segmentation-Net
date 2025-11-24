import torch
import torch.nn as nn

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Network(nn.Module):
    def __init__(
        self,
        input_shape=(3, 128, 128),
        num_classes=5,
        dropout=0.1,
    ):
        super().__init__()

        # Encoder
        self.conv1 = DoubleConv(3, 16, dropout=dropout)
        self.pool1 = nn.MaxPool2d(2)  # H/2
        self.conv2 = DoubleConv(16, 32, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)  # H/4
        self.conv3 = DoubleConv(32, 64, dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)  # H/8
        self.conv4 = DoubleConv(64, 128, dropout=dropout)

        # Decoder
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # H/8 → H/4
        self.conv5 = DoubleConv(128, 64, dropout=dropout)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # H/4 → H/2
        self.conv6 = DoubleConv(64, 32, dropout=dropout)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # H
        self.conv7 = DoubleConv(32, 16, dropout=dropout)
        # Final output
        self.out = nn.Conv2d(16, 5, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)  # 3 → 16
        x2 = self.conv2(self.pool1(x1))  # 16 → 32
        x3 = self.conv3(self.pool2(x2))  # 32 → 64
        x4 = self.conv4(self.pool3(x3))  # 64 → 128

        # Decoder
        y3 = self.up3(x4)  # H/8 → H/4
        y3 = torch.cat([y3, x3], dim=1)  # 64+64 = 128
        y3 = self.conv5(y3)

        y2 = self.up2(y3)  # H/4 → H/2
        y2 = torch.cat([y2, x2], dim=1)  # 32+32 = 64
        y2 = self.conv6(y2)

        y1 = self.up1(y2)  # H/2 → H
        y1 = torch.cat([y1, x1], dim=1)  # 16+16 = 32
        y1 = self.conv7(y1)

        return self.out(y1)

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
