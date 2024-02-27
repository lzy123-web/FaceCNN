from dataset_CK import FaceDataset
import Net
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision import transforms
import win32ui


def test():
    dlg = win32ui.CreateFileDialog(1)
    dlg.SetOFNInitialDir(r'D:\PCprojects\FaceCNN\data')
    dlg.DoModal()
    img_name = dlg.GetPathName()
    tfs = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])
    model = Net.MobileNetV2WithAttention(num_classes=7)  # 实例化一个网络
    model.cuda()  # 送入GPU，利用GPU计算
    if os.path.exists("./save/weight_MobileNetResnetWithAttention_CK.pt"):
        model.load_state_dict(torch.load("./save/weight_MobileNetResnetWithAttention_CK.pt")["MobileNetResnetWithAttention"])  # 加载训练好的模型参数
    model.eval()  # 设定为评估模式，即计算过程中不要dropout

    img = Image.open(img_name)



    img = img.convert('L')
    img = tfs(img)
    img = Variable(img).cuda().unsqueeze(0)
    out = model(img)
    out = torch.sigmoid(out)
    emotion_arg = np.argmax(out.cpu().detach().numpy())
    print(emotion_arg)
    emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    print(emotions[emotion_arg])


if __name__ == '__main__':
    test()
