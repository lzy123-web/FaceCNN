import common
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
from statistics import mode
from DBFace import DBFace
import Net

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


def detect_image(model, file):
    image = common.imread(file)
    objs = detect(model, image)

    for obj in objs:
        common.drawbbox(image, obj)

    common.imwrite("detect_result/" + common.file_name_no_suffix(file) + ".draw.jpg", image)


def image_demo():
    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()

    dbface.load("model/dbface.pth")
    detect_image(dbface, "datas/1.jpg")
    # detect_image(dbface, "datas/12_Group_Group_12_Group_Group_12_728.jpg")


def camera_demo():
    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()
    dbface.load("model/dbface.pth")
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./video/test.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()

    while ok:
        objs = detect(dbface, frame)

        for obj in objs:
            common.drawbbox(frame, obj)

        cv2.imshow("demo DBFace", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        ok, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


class Test:
    def __init__(self, path):
        self.net = Net.FaceCNN()
        self.dection_faceNet = DBFace().cuda()
        self.emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
        self.model_path = path

    # 人脸数据归一化,将像素值从0-255映射到0-1之间
    def preprocess_input(self, images):
        images = images / 255.0
        return images

    def gaussian_weights_init(self, m):
        classname = m.__class__.__name__
        # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.04)

    def camera_decetion(self, color=None):
        self.dection_faceNet.eval()
        self.dection_faceNet.load("../model/dbface.pth")
        self.net.eval()
        self.net.load_state_dict(torch.load(self.model_path)["faceCNN"])
        frame_window = 10
        emotion_window = []
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cap = cv2.VideoCapture(0)
        capture = cv2.VideoCapture("../video/test.mp4")
        ok, frame = capture.read()
        frame = frame[:, ::-1, :]  # 水平翻转，符合自拍习惯
        frame = frame.copy()
        # 获得灰度图，并且在内存中创建一个图像对象
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        while ok:
            objs = detect(self.dection_faceNet, frame)
            for obj in objs:
                if color is None:
                    color = common.randcolor(obj.label)
                x, y, r, b = common.intv(obj.box)
                w = r - x + 1
                h = b - y + 1
                cv2.rectangle(frame, (x, y, r - x + 1, b - y + 1), color, 2, 16)

                # border = thickness / 2
                # cv2.rectangle(frame, common.intv(x - border, y - 21, w + 2, 21), color, -1, 16)
                # cv2.putText(frame, text, pos, 0, 0.5, textcolor, 1, 16)
                face = gray[y:y + h, x:x + w]
                try:
                    # shape变为(48,48)
                    face = cv2.resize(face, (48, 48))
                except:
                    continue
                face = np.expand_dims(face, 0)
                face = np.expand_dims(face, 0)
                face = self.preprocess_input(face)
                new_face = torch.from_numpy(face)
                new_new_face = new_face.float().requires_grad_(False)
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

                cv2.putText(frame, emotion_mode, (x, y - 30), font, .7, (0, 0, 255), 1, cv2.LINE_AA)
            try:
                    # 将图片从内存中显示到屏幕上
                cv2.imshow('window_frame', frame)
            except:
                continue

                # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ok, frame = capture.read()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test = Test("../model/weight_FaceCNN.pt")
    test.camera_decetion()

    # image_demo()
    # camera_demo()
    # net = DBFace()
    # net.load("../model/dbface.pth")
    # print(net)
