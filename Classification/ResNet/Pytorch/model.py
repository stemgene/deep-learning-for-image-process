import torch
import torch.nn as nn

"""
首先定义残差结构，18-layer和34-layer采用一种残差结构，50-layer、101-layer和152-layer采用另一种
"""

# 18-layer和34-layer采用的残差结构
class BasicBlock(nn.Module):
    expansion = 1
    """
    in_channel 输入特征矩阵的深度，
    out_channel 输出特征矩阵的深度，对应主分支上卷积核的个数，
    stride=1 不改变高宽，对应实线的残差结构。stride=2，对应虚线的残差结构
    downsample对应的是虚线的残差结构，downsample的值不是boolean，而是通过sequential把kernel=1的卷积层和BN组合起来的小型网络
    """
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False) # 使用bn时，不使用bias
        self.bn1 = nn.BatchNorm2d(out_channel) # output = (input - 3 + 2*1)/2 + 1 = input / 2 + 0.5 = 向下取整(input/2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x # short cut分支上的输出值
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

# 定义50-layer、101-layer和152-layer的残差结构
class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4 # 第三层卷积层的个数是前两层的4倍
    
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=1, bias=False) # squeeze channel
        self.bn1 = nn.BatchNorm2d(out_channel) # output = (input - 3 + 2*1)/2 + 1 = input / 2 + 0.5 = 向下取整(input/2)
        # --------------------------------------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # --------------------------------------------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, padding=1, bias=False) # squeeze channel
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
        
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(x)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    # block即残差结构，block=BasicBlock即18、34层；block=Bottleneck即50、101、152层
    # blocks_num残差结构的数目，list参数，如34层的参数为[3, 4, 6, 3]代表每一个stage上需要多少个残差结构
    # num_class=1000, 即1000个分类
    # include_top 为了方便以后在ResNet基础上搭建更为复杂的网络
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # conv2_x ~ conv5_x
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            # 平均池化下采样
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 卷积层初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    # channel是残差结构中第一层卷积核的个数
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # 虚线short cut，对于18、34层会直接跳过if，对于50、101、152的网络，则会进入到此下采样的阶段。
        if stride != 1 or self.in_channel != channel * block.expansion:
            print("downsample input = ", self.in_channel, channel*block.expansion)
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        
        layers = []
        """
        in_channel, 上一层特征矩阵的深度
        channel，主分支上第一个卷积核的个数
        """
        print("first layer channel = ", self.in_channel)
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        # 通过虚线残差结构之后的卷积核个数，如深度网络第三层的4倍
        # 否则第一层输入的总是64
        self.in_channel = channel * block.expansion
        
        # 通过for loop构建实线残差部分，不需要下采样
        for _ in range(1, block_num):
            # self.in_channel: 输入特征矩阵的channel。channel：主线分支上第一个卷积核个数
            layers.append(block(self.in_channel, channel))
        
        return nn.Sequential(*layers) # *的意思是非关键字参数
    
    def forward(self, x):
        print('x: ', x.shape)
        x = self.conv1(x)
        print("after conv1 = ", x.shape)
        x = self.bn1(x)
        print("after bn = ", x.shape)
        x = self.relu(x)
        print("after relu = ", x.shape)
        x = self.maxpool(x)
        print("after maxpool = ", x.shape)
        
        x = self.layer1(x)
        print("after stage 1 = ", x.shape)
        x = self.layer2(x)
        print("after stage 2 = ", x.shape)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        
        return x

def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)