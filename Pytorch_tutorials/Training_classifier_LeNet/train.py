import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

transform = transforms.Compose(
    [transforms.ToTensor(), #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # z-score = (x - mu) / sigma, i.e. output = (input - 0.5) / 0.5
)

# 50000张训练图片
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=36, shuffle=True, num_workers=0)

# 10000张训练图片
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)

# get some random testing images
test_data_iter = iter(testloader)
test_image, test_label = test_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

start = time.time()
for epoch in range(5):
    running_loss = 0.0
    # 通过循环建立训练集样本
    for step, data in enumerate(trainloader, start=0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step() # update parameters
        
        # print statistics
        running_loss += loss.item()
        if step % 500 == 499:    # print every 500 mini-batches
            with torch.no_grad():  # 不要计算结点的误差损失梯度，节省内存和运算时间
                outputs = net(test_image) # [batch, 10]
                predict_y = torch.max(outputs, dim=1)[1]  #预测最大可能性 dim=0是batch
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)  # item()是得到tensor中的元素值
                
                print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f' %
                    (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

end = time.time()
print("Finished Training. It cost ", (start-end)/60)

save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)


# # show image
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize input = output * 0.5 + 0.5 = out / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0))) # to [channel, height, width]
#     plt.show()

# 需要更新以下testloader的batchsize=4
# testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
# print(" ".join('%5s' % classes[test_label[j]] for j in range(4)))
# imshow(torchvision.utils.make_grid(test_image))
