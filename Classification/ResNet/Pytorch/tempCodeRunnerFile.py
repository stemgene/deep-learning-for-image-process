    # model_weight_path = "resnet34-pre.pth" # if data_root is current directory of train.py
    # model_weight_path = data_root + "/Classification/ResNet/Pytorch/resnet34-pre.pth"  # if data_root is deep-learning-for-image-process
    # # 载入模型权重
    # missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
    # # for param in net.parameters():
    # #     param.requires_grad = False
    # # change fc layer structure 重新赋值全连接层的类别数
    # inchannel = net.fc.in_features
    # net.fc = nn.Linear(inchannel, 5)