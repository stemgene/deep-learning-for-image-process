import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 数据量小，只用一半的网络
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),     # input = [3, 224, 224]。对于原网络的padding = [1, 2]，可以用padding = (1, 2)来作为参数
            nn.ReLU(inplace=True),                                     # output = [48, 55, 55]
            nn.MaxPool2d(kernel_size=3, stride=2),                     # output = [48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),              # output = [128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                     # output = [192, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),             # output = [192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),             # output = [192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),             # output = [128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)                      # output = [128, 6, 6]
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        # 
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():  # 遍历每一个层机构
            # 如果是卷积层，用kaiming normal
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 如果bias不为空，用0初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 如果是fully connection，用正态分布
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) # mean=0, variance=0.01
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1) # 第0维是batch，从第1维开始是size
        x = self.classifier(x)
        return x