import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import type_of_target
from torch.utils.data import DataLoader
import dataset_CK
import Net
from sklearn.utils.multiclass import type_of_target


def draw_confusion_matrix(label_true, label_pred, label_name, normlize, title="Confusion Matrix", pdf_save_path=None,
                          dpi=100):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param normlize: 是否设元素为百分比形式
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          normlize=True,
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(label_true, label_pred)
    if normlize:
        row_sums = np.sum(cm, axis=1)  # 计算每行的和
        cm = cm / row_sums[:, np.newaxis]  # 广播计算每个元素占比

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[i, j]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)


def mateix(test_loader, net):
    y_gt = []
    y_pred = []
    for index, (imgs, labels) in enumerate(test_loader):
        labels_pd = net(imgs.cuda())
        predict_np = np.argmax(labels_pd.cpu().detach().numpy(), axis=-1)  # array([0,5,1,6,3,...],dtype=int64)
        labels_np = np.argmax(labels.numpy().astype(np.int64), axis=-1)  # array([0,5,0,6,2,...],dtype=int64)
        y_pred.extend(predict_np.tolist())
        y_gt.extend(labels_np.tolist())

    draw_confusion_matrix(label_true=np.array(y_gt),  # y_gt=[0,5,1,6,3,...]
                          label_pred=np.array(y_pred),  # y_pred=[0,5,1,6,3,...]
                          label_name=["An", "Di", "Fe", "Ha", "Sa", "Su", "Ne"],
                          normlize=True,
                          title="Confusion Matrix on CK+",
                          pdf_save_path="./confusion_matrix_fig/Confusion_Matrix_on_CK+_SE_MobileNet.jpg",
                          dpi=300)


def main(val_path):
    val_loader = DataLoader(dataset_CK.FaceDataset(val_path), batch_size=16, shuffle=False)
    mateix(val_loader, Net.MobileNetV2WithAttention(num_classes=7).cuda())


if __name__ == "__main__":
    main('./data/val')
