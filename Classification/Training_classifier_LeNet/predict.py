import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
net.load_state_dict(torch.load('Lenet.pth'))

im = Image.open('4.jpg')
im = transform(im) # [C, H, W]
im = torch.unsqueeze(im, dim=0) # dim=0在最前面增加一个新的维度 [N, C, H, W]

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].data.numpy()
print(classes[int(predict)])

# with softmax
# with torch.no_grad():
#     outputs = net(im)
#     predict = torch.softmax(outputs, dim=1)
# print(predict)
"""
tensor([[1.2167e-04, 7.0688e-05, 1.2883e-02, 1.0446e-02, 9.6217e-01, 3.2258e-03,
         6.5042e-03, 4.3587e-03, 1.2347e-05, 2.0690e-04]])
"""