import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from model import resnet34, resnet101

device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
print(data_root)
image_path = data_root + "/data_set/flower_data"  # flower data set path
print(image_path)