import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class UNetPP(nn.Module):
    """
    UNet++ (Nested U-Net) segmentation backbone.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=5,
        base_channels=32,
        dropout=0.1,
        deep_supervision=False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Channel numbers per depth
        nb_filter = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

        self.pool = nn.MaxPool2d(2, 2)

        def make_up(in_ch, out_ch):
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        # Encoder (x_0,0 ~ x_4,0)
        self.conv0_0 = DoubleConv(in_channels, nb_filter[0], dropout=0.0)
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1], dropout=0.5 * dropout)
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2], dropout=dropout)
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3], dropout=dropout)
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4], dropout=dropout)

        # x_0,1
        self.up1_0 = make_up(nb_filter[1], nb_filter[0])
        self.conv0_1 = DoubleConv(
            nb_filter[0] * 2,
            nb_filter[0],
            dropout=0.5 * dropout,
        )

        # x_1,1 & x_0,2
        self.up2_0 = make_up(nb_filter[2], nb_filter[1])
        self.conv1_1 = DoubleConv(
            nb_filter[1] * 2,
            nb_filter[1],
            dropout=dropout,
        )
        self.up1_1 = make_up(nb_filter[1], nb_filter[0])
        self.conv0_2 = DoubleConv(
            nb_filter[0] * 3,
            nb_filter[0],
            dropout=0.5 * dropout,
        )

        # x_2,1 & x_1,2 & x_0,3
        self.up3_0 = make_up(nb_filter[3], nb_filter[2])
        self.conv2_1 = DoubleConv(
            nb_filter[2] * 2,
            nb_filter[2],
            dropout=dropout,
        )
        self.up2_1 = make_up(nb_filter[2], nb_filter[1])
        self.conv1_2 = DoubleConv(
            nb_filter[1] * 3,
            nb_filter[1],
            dropout=dropout,
        )
        self.up1_2 = make_up(nb_filter[1], nb_filter[0])
        self.conv0_3 = DoubleConv(
            nb_filter[0] * 4,
            nb_filter[0],
            dropout=0.5 * dropout,
        )

        # x_3,1 & x_2,2 & x_1,3 & x_0,4
        self.up4_0 = make_up(nb_filter[4], nb_filter[3])
        self.conv3_1 = DoubleConv(
            nb_filter[3] * 2,
            nb_filter[3],
            dropout=dropout,
        )
        self.up3_1 = make_up(nb_filter[3], nb_filter[2])
        self.conv2_2 = DoubleConv(
            nb_filter[2] * 3,
            nb_filter[2],
            dropout=dropout,
        )
        self.up2_2 = make_up(nb_filter[2], nb_filter[1])
        self.conv1_3 = DoubleConv(
            nb_filter[1] * 4,
            nb_filter[1],
            dropout=dropout,
        )
        self.up1_3 = make_up(nb_filter[1], nb_filter[0])
        self.conv0_4 = DoubleConv(
            nb_filter[0] * 5,
            nb_filter[0],
            dropout=0.5 * dropout,
        )

        # Final classifier(s)
        if deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Nested decoder
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], dim=1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], dim=1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], dim=1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], dim=1)
        )

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return [
                F.softmax(out1, dim=1),
                F.softmax(out2, dim=1),
                F.softmax(out3, dim=1),
                F.softmax(out4, dim=1),
            ]
        else:
            out = self.final(x0_4)
            return F.softmax(out, dim=1)


class Network(nn.Module):
    """
    Wrapper to keep the same external interface as the original file.
    """

    def __init__(
        self,
        input_shape=(3, 128, 128),
        num_classes=5,
        dropout=0.1,
    ):
        super().__init__()
        in_channels = input_shape[0]
        self.model = UNetPP(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=32,  # ~9M params, close to original ~8.3M
            dropout=dropout,
            deep_supervision=False,
        )

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))

    def get_parameter(self):
        # 保持接口一致
        return {}


if __name__ == "__main__":
    input_shape = (3, 256, 256)
    num_classes = 5
    model = Network(input_shape, num_classes)
    x = torch.randn(1, *input_shape)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
