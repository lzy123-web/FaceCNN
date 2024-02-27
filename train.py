import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import Net
import dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
import os
import torch.nn.functional as F


class Trainer:
    record = {"train_loss": [], "val_loss": [], "acc": []}
    x_epoch = []

    def __init__(self, args, net_num):
        self.args = args
        self.device = self.args.device
        self.num_classes = 7
        self.net_num = net_num
        self.net_name = ["VGG", "Resnet", "MobileNet", "VGGWithAttention", "ResnetWithAttention",
                         "MobileNetWithAttention"]
        self.net_list = [Net.Vgg16(num_classes=self.num_classes),
                         Net.Resnet50(num_classes=self.num_classes),
                         Net.MobileNetV2(num_classes=self.num_classes),
                         Net.Vgg16WithAttention(num_classes=self.num_classes),
                         Net.Senet(),
                         Net.MobileNetV2WithAttention(num_classes=self.num_classes)
                         ]
        self.net = self.net_list[self.net_num]
        self.train_loader = DataLoader(dataset.FaceDataset(self.args.train_path),
                                       batch_size=self.args.batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset.FaceDataset(self.args.val_path),
                                     batch_size=self.args.batch_size, shuffle=False)
        self.criterion = nn.BCEWithLogitsLoss()       #BCE，BCEWithLogitsLoss
        self.epoch = 0
        self.lr = 1e-3
        if self.args.resume:
            if not os.path.exists(self.args.save_path):
                print("No params, start training...")
            else:
                params_dict = torch.load(self.args.save_path)
                self.epoch = params_dict["epoch"]
                self.net.load_state_dict(params_dict[self.net_name[self.net_num]], False)
                print("Loaded params from {}\n[Epoch]: {}   [lr]: {}".format(self.args.save_path,
                                                                             self.epoch,
                                                                             self.lr))
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)           # Adam

    def train(self, epoch):
        self.net.train()
        train_loss = 0.
        train_loss_all = 0.
        total = 0
        start_time = time.time()
        print("Start epoch: {}".format(epoch))

        for i, (img, label) in enumerate(self.train_loader):
            img = img.to(self.device)
            label = label.to(self.device)
            # out = self.net(img).view(self.args.batch_size, -1)
            out = self.net(img)
            loss = self.criterion(out, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_loss_all += loss.item()
            total += 1

            if (i + 1) % self.args.interval == 0:
                end_time = time.time()
                print("[Epoch]: {}[Progress: {:.1f}%]time:{:.2f} net_loss:{:.5f}".format(
                    epoch, (i + 1) * 100 / len(self.train_loader), end_time - start_time,
                           train_loss / self.args.interval,
                ))
                train_loss = 0.

        print("Save params to {}".format(self.args.save_path))
        param_dict = {
            "epoch": epoch,
            "lr": self.lr,
            self.net_name[self.net_num]: self.net.state_dict(),
        }
        torch.save(param_dict, self.args.save_path)
        return train_loss_all / len(self.train_loader)

    def val(self, epoch):
        self.net.eval()
        val_loss = 0.
        result = 0.
        count, total = 0, 0
        print("Test Start!")
        start_time = time.time()
        with torch.no_grad():
            for i, (img, label) in enumerate(self.val_loader):
                print(img.shape)
                img = img.to(self.device)
                label = label.to(self.device)
                # print(label)
                out = self.net(img)
                # out = self.net(img).view(self.args.batch_size, -1)
                loss = self.criterion(out, label)
                val_loss += loss.item()
                outs = torch.sigmoid(out)
                # print(outs)
                for out1 in outs:
                    emotion_arg = np.argmax(out1.cpu().detach().numpy())
                    for i in range(7):
                        if i == emotion_arg:
                            out1[i] = 1
                        else:
                            out1[i] = 0
                # print(outs)
                # out[out >= 0.95] = 1
                # out[out < 0.95] = 0
                # print(out)
                result += torch.mean(torch.sum(outs == label, dim=1) / outs.shape[1])
                # print(torch.mean(torch.sum(outs == label, dim=1) / outs.shape[1]))
                total += 1
                count += 1
            end_time = time.time()
            print("Test Finish!")
            print("[Epoch]: {} time:{:.2f} acc:{:.5f} loss: {:.5f}".format(
                epoch, end_time - start_time, result / total, val_loss / count
            ))
        return result / total, val_loss / count

    def draw_curve(self, fig, epoch, train_loss, val_loss):
        ax0 = fig.add_subplot(111, title="loss")
        self.record["train_loss"].append(train_loss)
        self.record["val_loss"].append(val_loss)
        # self.record["acc"].append(acc)
        self.x_epoch.append(epoch)

        ax0.plot(self.x_epoch, self.record["train_loss"], "b.-", label="train_loss")
        ax0.plot(self.x_epoch, self.record["val_loss"], "r.-", label="val_loss")
        # ax0.plot(self.x_epoch, self.record["acc"], "g.-", label="acc")
        if epoch == self.epoch:
            ax0.legend()
        if not os.path.exists(r"./train_fig"):
            os.makedirs(r"./train_fig")
        fig.savefig(r"./train_fig/train_loss_{}.jpg".format(self.net_name[self.net_num]))

    def draw_acc(self, acc):
        fig = plt.figure()
        acc = acc.cpu()
        self.record["acc"].append(acc)
        plt.plot(self.x_epoch, self.record["acc"], "g.-")
        fig.savefig(r"./train_fig/train_acc_{}.jpg".format(self.net_name[self.net_num]))


def main(args):
    t = Trainer(args=args, net_num=3)
    fig = plt.figure()
    for epoch in range(t.epoch, t.epoch + args.num_epochs):
        train_loss = t.train(epoch)
        acc, val_loss = t.val(epoch)
        t.draw_curve(fig, epoch, train_loss, val_loss)
        t.draw_acc(acc)
    file = h5py.File('./train_fig/{}.h5'.format(t.net_name[t.net_num], 'w'))
    file['train_loss'] = torch.tensor(t.record["train_loss"])
    file['val_loss'] = torch.tensor(t.record["val_loss"])
    file['acc'] = torch.tensor(t.record["acc"])
    file.close()


if __name__ == "__main__":
    net_name = ["VGG", "Resnet", "MobileNet", "VGGWithAttention", "ResnetWithAttention",
                "MobileNetWithAttention"]
    parser = argparse.ArgumentParser(description="Training {}".format(net_name[3]))
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--train_path", default=r"./image/train", type=str)
    parser.add_argument("--val_path", default=r"./image/test", type=str)
    parser.add_argument("--resume", default=False, type=bool)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--save_path", default=r"./save/weight_{}.pt".format(net_name[3]), type=str)
    parser.add_argument("--interval", default=70, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    if not os.path.exists(r"./save"):
        os.makedirs(r"./save")
    args1 = parser.parse_args()
    main(args1)
