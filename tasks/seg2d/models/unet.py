from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, nonlinearity=nn.ReLU, dropout=0):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nonlinearity(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nonlinearity(inplace=True)
        ]
        if dropout>0:
            layers.insert(3, nn.Dropout2d(dropout))
            layers.append(nn.Dropout2d(dropout))
        super(DoubleConv, self).__init__(*layers)


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, nonlinearity=nn.ReLU, dropout=0.5):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels, nonlinearity, dropout)
        )


class Down_Concat_DConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_Concat_DConv, self).__init__()
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_down, x_side):
        x_down = self.max_pool(x_down)
        x_concat = torch.cat([x_down, x_side], dim=1)
        x = self.double_conv(x_concat)
        return x


class Up(nn.Module):
    def __init__(self, in_channels_below, in_channels_skip, bilinear=True, nonlinearity=nn.ReLU, dropout=0.5):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels_below+in_channels_skip, in_channels_skip, nonlinearity, dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels_below, in_channels_skip, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels_skip*2, in_channels_skip, nonlinearity, dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    # 定义支持的文件扩展名
    supported_nonlinearity = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU
    }
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 4,
                 bilinear: bool = False,
                 layers: List[int] = [32, 64, 128, 256, 512, 512, 512],
                 deep_supervision: bool = False,
                 nonlinearity='ReLU',
                 dropout: float = 0.5):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision
        self.in_conv = DoubleConv(in_channels, layers[0]) # 1->32
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.seg_layers = nn.ModuleList()

        # 32 -> 64 -> 128 -> 256 -> 512#1 -> 512#2 -> 512#3
        for i in range(len(layers) - 1):
            p = 0
            if i >= len(layers)-4: p = dropout
            self.downs.append(Down(layers[i], layers[i + 1],
                                   UNet.supported_nonlinearity[nonlinearity],
                                   dropout=p))

        # 512#3 -> 512#2 -> 512#1 -> 256 -> 128 -> 64 -> 32
        for i in range(len(layers) - 1, 0, -1):
            p = 0
            if i >= len(layers)-2: p = dropout
            self.ups.append(Up(layers[i], layers[i - 1], bilinear, 
                               UNet.supported_nonlinearity[nonlinearity],
                               dropout=p))
            self.seg_layers.append(OutConv(layers[i-1], num_classes))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        downs_features = [x1]
        for down in self.downs:
            downs_features.append(down(downs_features[-1]))

        x = downs_features[-1]
        deep_outputs = []
        for i in range(len(self.ups)):
            x = self.ups[i](x, downs_features[-(i + 2)])
            if self.deep_supervision:
                deep_outputs.append(self.seg_layers[i](x))
            elif i == len(self.ups) - 1:
                deep_outputs.append(self.seg_layers[i](x))
        
        # invert seg outputs so that the largest segmentation prediction is returned first
        deep_outputs = deep_outputs[::-1]

        if self.deep_supervision and self.training:
            r = deep_outputs
        else:
            r = deep_outputs[0]
        return r

class nnUNet(nn.Module):
    def __init__(self, in_channels: int = 1,
                 num_classes: int = 4,
                 path=None, 
                 layers: List[int] = [32, 64, 128, 256, 512, 512, 512],
                 deep_supervision=False):
        super(nnUNet, self).__init__()
        self.model = torch.load(path)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    net = UNet(in_channels=1, num_classes=4, bilinear=False, layers=[32, 64, 128, 256, 512, 512, 512],
               deep_supervision=True, nonlinearity='LeakyReLU', dropout=0.5)
    x = torch.randn((1, 1, 320, 320))
    out = net(x)
    for i in out:
        print(i.shape)