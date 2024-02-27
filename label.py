import cv2 as cv
import os

# 生成ck+ label.txt

# if not os.path.exists("./data"):
#     os.makedirs("./data")
emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
emotions_label = [i for i in range(7)]
# print(label)
file_path = "./data/val"
file_list = os.listdir(file_path)
print(file_list)


def labeled(path):
    with open(path, 'w') as file_obj:
        for i in range(len(file_list)):
            img_path = os.path.join(file_path + "/" + file_list[i])
            img_list = os.listdir(img_path)
            for img in img_list:
                file_name = os.path.join(img_path, img)
                label = emotions_label[i]
                file_obj.write("{} {}".format(file_name, label))
                file_obj.write("\n")
            # print(img_list)


if __name__ == '__main__':
    labeled("./data/val/label.txt")
    # labeled("./data/val/label.txt")
    # list = []
    # with open("./data/label.txt", "r") as f:  # 打开文件
    #     for line in f.readlines():
    #         line = line.strip('\n')
    #         list.append(line)
    # print(len(list))