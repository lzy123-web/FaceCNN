import pandas as pd
import cv2
import numpy as np
import os


class DataProcess():
    def __init__(self):
        self.train_path = r'J:\FaceCNN\train.csv'
        self.label_path = r'./data/label.csv'
        self.data_path = r'./data/data.csv'
        self.image_path = './image1'  # 指定存放图片的路径

    def process(self):
        # 数据预处理
        # 将label与人脸数据作拆分
        df = pd.read_csv(self.train_path)  # pd阅读器打开csv文件
        df = df.fillna(0)  # 空值填充

        # 分别提取标签和特征数据
        df_y = df[['label']]
        df_x = df[['feature']]

        # 将label,feature数据写入csv文件
        df_y.to_csv(self.label_path, index=False, header=False)  # 不保存索引(0-N),不保存列名('label')
        df_x.to_csv(self.data_path, index=False, header=False)

        # 读取像素数据
        data = np.loadtxt(self.data_path)

        # 按行取数据
        for i in range(data.shape[0]):  # 按行读取
            face_array = data[i, :].reshape((48, 48))  # reshape 转成图像矩阵给cv2处理
            cv2.imwrite(self.image_path + '//' + '{0}.jpg'.format(i), face_array)  # csv文件转jpg写图片

    def data_label(self, path):
        df_label = pd.read_csv(self.label_path, header=None)
        files_dir = os.listdir(path)
        path_list = []
        label_path = []

        for file in files_dir:
            if os.path.splitext(file)[1] == ".jpg":
                path_list.append(file)
                index = int(os.path.splitext(file)[0])
                label_path.append(df_label.iat[index, 0])

        path_s = pd.Series(path_list)
        label_s = pd.Series(label_path)
        df = pd.DataFrame()
        df["path"] = path_s
        df["label"] = label_s
        df.to_csv(path + '\\dataset.csv', index=False, header=False)


if __name__ == "__main__":
    d = DataProcess()
    d.process()
    # d.data_label(r'./image/train')
    # d.data_label(r'./image/test')
