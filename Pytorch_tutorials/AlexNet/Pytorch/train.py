import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import time
from model import AlexNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
}

# get current path
current_path1 = os.getcwd()
current_path2 = os.path.abspath(os.getcwd())

# get data root path
data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
image_path = data_root + "/data_set/flower_data/"
# The same path
data_root1 = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
image_path1 = data_root1 + "/data_set/flower_data/"

train_dataset = datasets.ImageFolder(root=image_path + "/train", transform=data_transform["train"])
train_num = len(train_dataset)

# get flower category index
flower_list = train_dataset.class_to_idx 
#print(flower_list) #{'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflower': 3, 'tulips': 4}
# reverse key : value
cla_dict = dict((val, key) for key, val in flower_list.items())

# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
"""
{
    "0": "daisy",
    "1": "dandelion",
    "2": "roses",
    "3": "sunflowers",
    "4": "tulips"
}
"""
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
validate_dataset = datasets.ImageFolder(root=image_path + "/val", transform=data_transform['val'])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

# # preview image
# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize input = output * 0.5 + 0.5 = out / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0))) # to [channel, height, width]
#     plt.show()# 需要更新以下testloader的batchsize=4 and shuffle=True否则只读取同一种类
# # testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=0)
# print(" ".join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))

net = AlexNet(num_classes=5, init_weights=True)

net.to(device)
loss_function = nn.CrossEntropyLoss()
# 调试用，查看模型参数
#para = list(net.parameters())
optimizer = optim.Adam(net.parameters(), lr=0.0002)
save_path = './AlexNet.pth'
best_acc = 0.0

for epoch in range(10):
    # train 通过此方法可以实现在training过程中启用dropout，而在validation阶段不用
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    t2 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step() # update parameters
        
        # print statistics
        running_loss += loss.item()
        # print training process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(loss, a, b, int(rate*100)), end="")
    print()
    print(time.perf_counter() - t1)
    
    # validate
    net.eval()
    acc = 0.0 # accumulate accurate number / epoch
    with torch.no_grad(): # 不要计算结点的误差损失梯度，节省内存和运算时间
        for data_test in validate_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1] # 返回输出的最大值
            # 对比预测值和真实label
            acc += (predict_y == test_labels.to(device)).sum().item() 
        # 准确率 预测正确数量 / 所有数量
        accurate_test = acc / val_num
        # 保存best_acc
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f test_accuracy: %.3f time_cost: %.3f' %(epoch+1, running_loss/step, acc/val_num, (time.perf_counter() - t2)/60))

print("Finished Training")

