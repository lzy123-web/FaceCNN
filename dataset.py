import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import cv2 as cv


class FaceDataset(data.Dataset):
    def __init__(self, root):
        super(FaceDataset, self).__init__()
        self.root = root
        df_path = pd.read_csv(root + '/dataset.csv', header=None, usecols=[0])
        df_label = pd.read_csv(root + '/dataset.csv', header=None, usecols=[1])
        # 将其中内容放入numpy, 方便后期索引
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]

    def __getitem__(self, item):
        face = cv.imread(self.root + "/" + self.path[item])
        face_gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
        face_hist = cv.equalizeHist(face_gray)
        # face_hist = cv.resize(face_hist, (224, 224))
        face_normalized = face_hist.reshape(1, 48, 48) / 255.0
        face_tensor = torch.from_numpy(face_normalized)
        face_tensor = face_tensor.type('torch.FloatTensor')
        label = torch.zeros([7])
        label[self.label[item]] = 1
        # label = torch.tensor(label)
        return face_tensor, label

    def __len__(self):
        return self.path.shape[0]


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    f = FaceDataset(r"./image/train")
    # print(f.label)
    a, b = f[4]
    # print(a)
    print(a.shape)
    print(b)
