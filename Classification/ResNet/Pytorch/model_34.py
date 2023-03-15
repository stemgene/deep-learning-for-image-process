import torch
import torch.nn as nn


# normal block
class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        self.test_in = in_channel
        self.test_out = out_channel
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        
        #self.initialize_weights()
    
    def forward(self, x):
        print("block in shape = ", x.shape)
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
            print("identity shape = ", identity.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        print("block out shape = ", out.shape)
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, block_num, num_classes=1000):
        super().__init__()
        #self.include_top = include_top
        self.in_channel = 64
        
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # output size = (1, 1)
        self.fc = nn.Linear(512, num_classes)
        
        # 卷积层初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1: # 虚线
            downsample = nn.Sequential(
                nn.Conv2d(int(channel / 2), channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel)
            )
        layers = []
        # 第一层
        print("first layer channel = ", self.in_channel)
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel
        # 第2-3层
        for _ in range(1, block_num):
            layers.append(block(channel, channel))
        
        return nn.Sequential(*layers)
            
        
    def forward(self, x):
        print('x: ', x.shape) # [1, 3, 244, 244]
        #print(x[0][0])
        x = self.conv1(x)
        # print("after conv1 = ", x.shape) # [1, 64, 112, 112]
        #print(x[0][0])
        x = self.bn1(x)
        #print(x[0][0])
        x = self.relu(x)
        x = self.maxpool(x)
        # print("after maxpool = ", x.shape) # [1, 64, 56, 56]
        
        # stages
        x = self.layer1(x)
        print("after stage 1 = ", x.shape)
        x = self.layer2(x)
        print("after stage 2 = ", x.shape)
        x = self.layer3(x)
        print("after stage 3 = ", x.shape)
        x = self.layer4(x)
        print("after stage 4 = ", x.shape)
        
        x = self.avgpool(x)
        print("after avgpool = ", x.shape) # [1, 512, 1, 1]
        x = torch.flatten(x, 1)
        print("After flatten = ", x.shape) # [1, 512]
        x = self.fc(x)
        print("after fc = ", x.shape) # [1, 1000]
        
        return x
    
def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])