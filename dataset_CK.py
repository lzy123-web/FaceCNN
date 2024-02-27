import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import cv2 as cv
from torchvision import transforms
import os
import ast
from PIL import Image


class FaceDataset(data.Dataset):
    def __init__(self, root):
        super(FaceDataset, self).__init__()
        self.root = root
        self.dataset = []
        self.tfs = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])
        with open(root + '/label.txt', "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.dataset.append(line)

    def __getitem__(self, item):
        img_path = self.dataset[item].split(" ")[0]
        label_idx = self.dataset[item].split(" ")[1]
        # label_idx = list(map(int, label))
        label = torch.zeros([7])
        label[int(label_idx)] = 1
        img = Image.open(img_path)
        img_gray = img.convert('L')
        img_gray = self.tfs(img_gray)
        return img_gray, label

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    # np.set_printoptions(threshold=np.inf)
    f = FaceDataset(r"./data/train")
    # print(f.label)
    a, b = f[1000]
    # print(a)
    print(type(a))
    print(b)
