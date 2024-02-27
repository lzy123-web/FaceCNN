import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# class MobileNetV2(nn.Module):
#     def __init__(self, num_classes):
#         super(MobileNetV2, self).__init__()
#         mobile_net_v2 = models.mobilenet_v2(True)
#         self.features_layer = mobile_net_v2.features
#         self.features_layer[0] = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
#             nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU6(inplace=True)
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.2, inplace=False),
#             nn.Linear(1280, num_classes)
#         )

#     def forward(self, x):
#         x = self.features_layer(x)
#         x= x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

class Block(nn.Module):
    def __init__(self, in_channels, filters, stride=1, is_1x1conv=False):
        super(Block, self).__init__()
        filter1, filter2, filter3 = filters
        self.is_1x1conv = is_1x1conv
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, filter1, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filter1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filter2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(filter2, filter3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(filter3),
        )
        if is_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filter3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filter3)
            )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(filter3, filter3 // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(filter3 // 16, filter3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_shortcut = x
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x2 = self.se(x1)
        x1 = x1 * x2
        if self.is_1x1conv:
            x_shortcut = self.shortcut(x_shortcut)
        x1 = x1 + x_shortcut
        x1 = self.relu(x1)
        return x1


class senet(nn.Module):
    def __init__(self, cfg):
        super(senet, self).__init__()
        classes = cfg['classes']
        num = cfg['num']
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(64, (64, 64, 256), num[0], 1)
        self.conv3 = self._make_layer(256, (128, 128, 512), num[1], 2)
        self.conv4 = self._make_layer(512, (256, 256, 1024), num[2], 2)
        self.conv5 = self._make_layer(1024, (512, 512, 2048), num[3], 2)
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_average_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layer(self, in_channels, filters, num, stride=1):
        layers = []
        block_1 = Block(in_channels, filters, stride=stride, is_1x1conv=True)
        layers.append(block_1)
        for i in range(1, num):
            layers.append(Block(filters[2], filters, stride=1, is_1x1conv=False))
        return nn.Sequential(*layers)


def Senet():
    cfg = {
        'num': (3, 4, 6, 3),
        'classes': (7)
    }
    return senet(cfg)



class Vgg16(nn.Module):
    def __init__(self, num_classes=1000):
        super(Vgg16, self).__init__()
        # 定义搭建网络的模块
        # 送入Conv2d的必须是四维tensor,[batch, channel, width, height]
        model = models.vgg16(pretrained=True)
        self.features = model.features
        self.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.avg_pool = model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Vgg16WithAttention(nn.Module):
    def __init__(self, num_classes=1000):
        super(Vgg16WithAttention, self).__init__()
        # 定义搭建网络的模块
        # 送入Conv2d的必须是四维tensor,[batch, channel, width, height]
        model = models.vgg16(pretrained=True)
        self.features = model.features
        self.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.att = SEBlock(512)
        self.avg_pool = model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.att(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Resnet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(Resnet50, self).__init__()
        model = models.resnet50(pretrained=True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                               bias=False)
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc(x)
        return x


class Resnet50WithAttention(nn.Module):
    def __init__(self, num_classes=1000):
        super(Resnet50WithAttention, self).__init__()
        model = models.resnet50(pretrained=True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                     bias=False)
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.att = Attention()
        self.avgpool = model.avgpool
        self.fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.att(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc(x)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        self.features = model.features
        self.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.classifier = model.classifier
        self.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# class MobileNetV2WithAttention(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(MobileNetV2WithAttention, self).__init__()
#         model = models.mobilenet_v2(pretrained=True)
#         self.features = model.features
#         self.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         self.att = Attention()
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         # self.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         self.classifier = model.classifier
#         self.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.att(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.layer1 = nn.Conv2d(2, 1, 3, 1, 1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.layer1(x1))


class SEBlock(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MobileNetV2WithAttention(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2WithAttention, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        self.features = model.features
        self.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.att = SEBlock(1280)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.classifier = model.classifier
        self.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.att(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # conv_arch = ((2, 1, 32), (3, 32, 64), (3, 64, 128))
    # # 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
    # fc_features = 128 * 6 * 6  # c * w * h
    # fc_hidden_units = 4096  # 任意

    print(models.vgg16())
    print(Vgg16(num_classes=7))
    # net = MobileNetV2WithAttention(num_classes=7)
    # print(net)
    # net2 = models.mobilenet_v2()
    # # print(net2)
    # x = torch.randn([1, 1, 48, 48])
    # y = net(x)
    # print(y.size())
