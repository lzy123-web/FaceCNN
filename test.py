import cv2 as cv
import torch
import numpy as np
from statistics import mode
import torch.nn.functional as F
import Net
import common
import DBFace

HAS_CUDA = torch.cuda.is_available()
print(f"HAS_CUDA = {HAS_CUDA}")


def nms(objs, iou=0.5):
    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


def detect(model, image, threshold=0.4, nms_iou=0.5):
    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]

    image = common.pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)

    torch_image = torch.from_numpy(image)[None]
    if HAS_CUDA:
        torch_image = torch_image.cuda()

    hm, box, landmark = model(torch_image)
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices // hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())
    box = box.cpu().squeeze().data.numpy()
    landmark = landmark.cpu().squeeze().data.numpy()

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (common.exp(x5y5 * 4) + ([cx] * 5 + [cy] * 5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
    return nms(objs, iou=nms_iou)


# 人脸数据归一化,将像素值从0-255映射到0-1之间
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


def preprocess_input(images):
    images = images / 255.0
    return images


class Test:
    def __init__(self, net_num):
        self.net_name = ["VGG", "Resnet", "MobileNet", "VGGWithAttention", "ResnetWithAttention", "MobileNet"]
        self.num_classes = 7
        self.net_num = net_num
        self.detection_model_path = r'./save/haarcascade_frontalface_default.xml'
        # self.net = Net.FaceCNN()
        self.net_list = [Net.Vgg16(num_classes=self.num_classes), Net.Resnet50(num_classes=self.num_classes),
                         Net.MobileNetV2(num_classes=self.num_classes),
                         Net.Vgg16WithAttention(num_classes=self.num_classes), Net.Senet(),
                         Net.MobileNetV2WithAttention(num_classes=self.num_classes)]
        self.model_path = [
            "./save/weight_VGG.pt", "./save/weight_Resnet.pt", "./save/weight_MobileNet.pt",
            "./save/weight_VGGWithAttention.pt", "./save/weight_ResnetWithAttention.pt",
            "./save/weight_MobileNetWithAttention.pt"
        ]
        self.net = self.net_list[self.net_num]
        self.decetion_faceNet = DBFace.DBFace().cuda()
        self.emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

    def detection(self, number):
        face_detection = cv.CascadeClassifier(self.detection_model_path)
        self.net.eval()
        self.net.load_state_dict(torch.load(self.model_path)[self.net_name[number]])
        frame_window = 10
        emotion_window = []
        # video_capture = cv.VideoCapture(0)
        # 视频文件识别
        video_capture = cv.VideoCapture("video/example_dsh.mp4")
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.startWindowThread()
        cv.namedWindow('window_frame')
        while True:
            _, frame = video_capture.read()
            # frame = frame[:, ::-1, :]  # 水平翻转，符合自拍习惯
            frame = frame.copy()
            # 获得灰度图，并且在内存中创建一个图像对象
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # 获取当前帧中的全部人脸
            faces = face_detection.detectMultiScale(gray, 1.3, 5)
            # 对于所有发现的人脸
            for (x, y, w, h) in faces:
                # 在脸周围画一个矩形框，(255,0,0)是颜色，2是线宽
                cv.rectangle(frame, (x, y), (x + w, y + h), (84, 255, 159), 2)

                # 获取人脸图像
                face = gray[y:y + h, x:x + w]

                try:
                    face = cv.resize(face, (48, 48))
                except:
                    continue

                # 扩充维度，shape变为(1,48,48,1)
                # 将（1，48，48，1）转换成为(1,1,48,48)
                face = np.expand_dims(face, 0)
                face = np.expand_dims(face, 0)

                # 人脸数据归一化，将像素值从0-255映射到0-1之间
                face = preprocess_input(face)
                new_face = torch.from_numpy(face)
                new_new_face = new_face.float().requires_grad_(False)

                # 调用我们训练好的表情识别模型，预测分类
                emotion_arg = np.argmax(self.net.forward(new_new_face).detach().numpy())
                emotion = self.emotion_labels[emotion_arg]

                emotion_window.append(emotion)

                if len(emotion_window) >= frame_window:
                    emotion_window.pop(0)

                try:
                    # 获得出现次数最多的分类
                    emotion_mode = mode(emotion_window)
                except:
                    continue

                # 在矩形框上部，输出分类文字
                cv.putText(frame, emotion_mode, (x, y - 30), font, .7, (0, 0, 255), 1, cv.LINE_AA)

            try:
                # 将图片从内存中显示到屏幕上
                cv.imshow('window_frame', frame)
            except:
                continue

            # 按q退出
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv.destroyAllWindows()

    def detectionWithDBface(self, model_num, color=None):
        self.decetion_faceNet.eval()
        self.decetion_faceNet.load("./model/dbface.pth")
        self.net.eval()
        self.net.load_state_dict(torch.load(self.model_path[model_num])[self.net_name[self.net_num]])
        frame_window = 10
        emotion_window = []
        font = cv.FONT_HERSHEY_SIMPLEX

        capture = cv.VideoCapture(0)
        # capture = cv.VideoCapture("./video/1.mp4")

        ok, frame = capture.read()     #获取视频中一帧图片
        frame = frame[:, ::-1, :]  # 水平翻转，符合自拍习惯
        frame = frame.copy()
        # 获得灰度图，并且在内存中创建一个图像对象
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        while ok:
            objs = detect(self.decetion_faceNet, frame)
            for obj in objs:
                if color is None:
                    color = common.randcolor(obj.label)
                x, y, r, b = common.intv(obj.box)
                w = r - x + 1
                h = b - y + 1
                cv.rectangle(frame, (x, y, r - x + 1, b - y + 1), color, 2, 16)

                # border = thickness / 2
                # cv2.rectangle(frame, common.intv(x - border, y - 21, w + 2, 21), color, -1, 16)
                # cv2.putText(frame, text, pos, 0, 0.5, textcolor, 1, 16)
                face = gray[y:y + h, x:x + w]
                try:
                    # shape变为(48,48)
                    face = cv.resize(face, (48, 48))
                except:
                    continue
                face = np.expand_dims(face, 0)
                face = np.expand_dims(face, 0)
                face = preprocess_input(face)
                new_face = torch.from_numpy(face)
                new_new_face = new_face.float().requires_grad_(False)
                emotion_arg = np.argmax(self.net(new_new_face).detach().numpy())
                emotion = self.emotion_labels[emotion_arg]
                emotion_window.append(emotion)

                if len(emotion_window) >= frame_window:
                    emotion_window.pop(0)
                try:
                    # 获得出现次数最多的分类
                    emotion_mode = mode(emotion_window)
                except:
                    continue

                cv.putText(frame, emotion_mode, (x, y - 30), font, .7, (0, 0, 255), 1, cv.LINE_AA)
            try:
                # 将图片从内存中显示到屏幕上
                cv.imshow('window_frame', frame)
            except:
                continue

                # 按q退出
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            ok, frame = capture.read()
        capture.release()
        cv.destroyAllWindows()


def main():
    """""
       model_path = [
           "./save/weight_VGG.pt", "./save/weight_Resnet.pt", "./save/weight_MobileNet.pt",
           "./save/weight_VGGWithAttention.pt", "./save/weight_ResnetWithAttention.pt",
           "./save/weight_MobileNetWithAttention.pt"
       ]
       """""
    test = Test(net_num=5)
    # test.detection()
    test.detectionWithDBface(model_num=5)


if __name__ == '__main__':
    main()


