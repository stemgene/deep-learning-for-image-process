import torch
from model import BottleNeckBlock
from torchinfo import summary
x = torch.rand(1, 32, 7, 7)
block = BottleNeckBlock(32, 64)
summary(block, input_size=(1, 32, 7 ,7))
print(block(x).shape) # [1, 64, 7, 7]
