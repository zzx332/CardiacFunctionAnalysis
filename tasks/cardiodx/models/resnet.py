import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 ori_c,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(ori_c, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet18(ori_c=3, num_classes=1000, include_top=True):
    """
    ResNet18实现
    """
    return ResNet(ori_c, BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

def resnet34(ori_c=3, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(ori_c, BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(ori_c=3, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(ori_c, Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

class TwoStreamResNet(nn.Module):
    """
    双流ResNet网络，共享权重
    专为小数据集设计
    """
    def __init__(self, 
                 in_channels: int = 3,
                 num_classes: int = 10,
                 backbone_type: str = 'resnet18',
                 fusion_method: str = 'concat',
                 include_top: bool = False):
        """
        Args:
            backbone_type: 骨干网络类型 'resnet34' 或 'resnet50'
            num_classes: 分类类别数
            fusion_method: 特征融合方法 'concat', 'sum', 'average'
            include_top: 是否包含原始分类头
        """
        super(TwoStreamResNet, self).__init__()
        
        self.fusion_method = fusion_method
        
        # 创建共享的backbone网络
        if backbone_type == 'resnet18':
            self.backbone = resnet18(ori_c=in_channels, num_classes=num_classes, include_top=include_top)
        elif backbone_type == 'resnet34':
            self.backbone = resnet34(ori_c=in_channels, num_classes=num_classes, include_top=include_top)
        elif backbone_type == 'resnet50':
            self.backbone = resnet50(ori_c=in_channels, num_classes=num_classes, include_top=include_top)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # 获取特征维度
        if backbone_type == 'resnet18' or backbone_type == 'resnet34':
            feature_dim = 512 * BasicBlock.expansion
        else:  # resnet50
            feature_dim = 512 * Bottleneck.expansion
        
        # 根据融合方法调整分类器输入维度
        if fusion_method == 'concat':
            classifier_input_dim = feature_dim * 2
        else:  # 'sum' or 'average'
            classifier_input_dim = feature_dim
        
        # 融合后的分类器 - 使用更简单的结构防止过拟合
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 增加dropout防止过拟合
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def extract_features(self, x):
        """提取特征，不包含最后的分类层"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x_a, x_b):
        """
        Args:
            x_a: 输入流A [batch_size, 3, H, W]
            x_b: 输入流B [batch_size, 3, H, W]
        Returns:
            分类结果 [batch_size, num_classes]
        """
        # 通过共享的backbone提取特征
        features_a = self.extract_features(x_a)  # [batch_size, feature_dim]
        features_b = self.extract_features(x_b)  # [batch_size, feature_dim]
        
        # 特征融合
        if self.fusion_method == 'concat':
            fused_features = torch.cat([features_a, features_b], dim=1)
        elif self.fusion_method == 'sum':
            fused_features = features_a + features_b
        elif self.fusion_method == 'average':
            fused_features = (features_a + features_b) / 2
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        
        # 分类
        output = self.classifier(fused_features)
        
        return output


class TwoStreamResNetWithEarlyFusion(nn.Module):
    """
    双流ResNet网络 - 早期融合版本
    在输入层就融合两个流
    """
    def __init__(self, 
                 backbone_type: str = 'resnet34',
                 num_classes: int = 10):
        """
        Args:
            backbone_type: 骨干网络类型 'resnet34' 或 'resnet50'
            num_classes: 分类类别数
        """
        super(TwoStreamResNetWithEarlyFusion, self).__init__()
        
        # 修改输入通道数为6（3+3）
        if backbone_type == 'resnet34':
            self.backbone = resnet34(ori_c=6, num_classes=num_classes, include_top=True)
        elif backbone_type == 'resnet50':
            self.backbone = resnet50(ori_c=6, num_classes=num_classes, include_top=True)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

    def forward(self, x_a, x_b):
        """
        Args:
            x_a: 输入流A [batch_size, 3, H, W]
            x_b: 输入流B [batch_size, 3, H, W]
        Returns:
            分类结果 [batch_size, num_classes]
        """
        # 在通道维度拼接两个输入
        x = torch.cat([x_a, x_b], dim=1)  # [batch_size, 6, H, W]
        
        # 通过backbone网络
        output = self.backbone(x)
        
        return output