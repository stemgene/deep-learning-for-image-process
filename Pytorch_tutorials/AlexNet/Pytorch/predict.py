import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
from model import AlexNet

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# load image
img = Image.open("../../tulip.jpg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# Read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# Create model
model = AlexNet(num_classes=5)
# load model weights
model_weight_path = './AlexNet.pth'
model.load_state_dict(torch.load(model_weight_path))
# 用eval模式，关闭dropout
model.eval() 
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img)) #数据正向传播得到output，再压缩
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)], predict[predict_cla].item())
plt.show()
