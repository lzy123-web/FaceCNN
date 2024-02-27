from dataset_CK import FaceDataset
from Net import Resnet50
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision import transforms


def test(img_path, model_file):
    tfs = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])
    model = Resnet50(num_classes=7).model  # 实例化一个网络
    model.cuda()  # 送入GPU，利用GPU计算
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file)["Resnet"])  # 加载训练好的模型参数
    model.eval()  # 设定为评估模式，即计算过程中不要dropout

    # datafile = FaceDataset(r"./data/val")
    # print('Dataset loaded! length of train set is {0}'.format(len(datafile)))
    img = Image.open(img_path)
    img = img.convert('L')
    img = tfs(img)
    img = Variable(img).cuda().unsqueeze(0)
    out = model(img)
    out = torch.sigmoid(out)
    emotion_arg = np.argmax(out.cpu().detach().numpy())
    print(emotion_arg)
    emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    print(emotions[emotion_arg])
    # out[out >= 0.95] = 1
    # out[out < 0.95] = 0

    # if out[0, 0] > out[0, 1]:  # 猫的概率大于狗
    #     print('the image is a cat')
    # else:  # 猫的概率小于狗
    #     print('the image is a dog')
    #
    # img = Image.open(datafile.list_img[index])  # 打开测试的图片
    # plt.figure('image')  # 利用matplotlib库显示图片
    # plt.imshow(img)
    # plt.show()


if __name__ == '__main__':
    test(r"F:\Spyder projects\Deep-Learning\FaceCNN\data\val\disgust\S130_012_00000009.png",
         r"./save/weight_Resnet_CK.pt")
