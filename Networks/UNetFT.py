import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class EfficientDiscriminativeFrequencyDomainBasedFFN(nn.Module):
    """
    EDFFN Module from https://doi.org/10.48550/arXiv.2405.14343
    A FFN module that leverages frequency domain learning for efficient feature.
    Hope to catch strong peridoic patterns in facade images like windows and balconies.
    """

    def __init__(self, dim=3, expansion_ratio=1 / 2, patch_size=4):
        super().__init__()
        hidden_features = int(dim * expansion_ratio)
        self.patch_size = patch_size

        self.in_proj1 = nn.Sequential(
            nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=False),
            nn.Conv2d(
                hidden_features * 2,
                hidden_features * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden_features * 2,
                bias=False,
            ),
        )
        self.gate_fn = nn.GELU()
        self.in_proj2 = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)
        self.w_freq = nn.Parameter(
            torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )

    def fft(self, x_patch):
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = self.w_freq * x_patch_fft
        return torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))

    def forward(self, x):
        # Gated conv-based FFN
        x_shortcut = x.clone()
        x1, x2 = self.in_proj1(x).chunk(2, dim=1)
        x = self.gate_fn(x1) * x2
        x = self.in_proj2(x)
        # FFT
        b, c, h, w = x.shape
        h_n = (self.patch_size - h % self.patch_size) % self.patch_size
        w_n = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, w_n, 0, h_n), mode="reflect")
        x_patch = rearrange(
            x,
            "b c (h patch1) (w patch2) -> b c h w patch1 patch2",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )
        x_patch = self.fft(x_patch)
        x = rearrange(
            x_patch,
            "b c h w patch1 patch2 -> b c (h patch1) (w patch2)",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )
        x = x[:, :, :h, :w]
        return x + x_shortcut


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

        # Fourier transform skip connection
        # self.fconv1 = DoubleFConv(32, dropout=dropout)
        # self.fconv2 = DoubleFConv(16, dropout=dropout)
        self.fconv1 = EfficientDiscriminativeFrequencyDomainBasedFFN(32)
        self.fconv2 = EfficientDiscriminativeFrequencyDomainBasedFFN(16)

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
        x2_f = self.fconv1(x2)
        y2 = torch.cat([y2, x2_f], dim=1)  # 32+32 = 64
        y2 = self.conv6(y2)

        y1 = self.up1(y2)  # H/2 → H
        x1_f = self.fconv2(x1)
        y1 = torch.cat([y1, x1_f], dim=1)  # 16+16 = 32
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
