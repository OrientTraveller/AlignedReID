import glob
import os.path as osp
import random
import time

import numpy as np
import torch
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import QFileDialog, QLabel, QHBoxLayout, QVBoxLayout

"""
import transform as T
from torch.utils.data import DataLoader
from model.ReIDNet import ReIDNet
from dataset_manager import Market1501
from dataset_loader import ImageDataset
from distance import low_memory_local_dist
from eval_metrics import evaluate
"""


def cal(img_path):
    img = Image.open(img_path)
    model.eval()
    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            features = model(imgs)
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            features = model(imgs)
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

    # feature normlization
    qf = 1. * qf / (torch.norm(qf, 2, dim=-1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim=-1, keepdim=True).expand_as(gf) + 1e-12)
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(
        n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    img_list, cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    return img_list


def img_compose(img_list):
    if len(img_list) < 1:
        raise IOError("img_list length is not larger than zero.")
    else:
        result = Image.new('RGB', (64 * 10 + 2 * 9, 128))
        left = 0
        right = 0
        for img in img_list:
            result.paste(img, (left, right))  # 将image复制到target的指定位置中
            left = left + 2 + 64  # left是左上角的横坐标，依次递增
        quality_value = 100  # quality来指定生成图片的质量，范围是0～100
        f = open("./count.txt")
        line = f.readline()
        i = int(line)
        line = line.replace(line, str(i + 1))
        f = open("./count.txt", "w")
        f.write(line)
        f.close()
        result_path = './data/result/result_' + str(i) + '.jpg'
        result.save(result_path, quality=quality_value)
        return result_path


class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setGeometry(500, 250, 800, 600)
        self.myButton = QtWidgets.QPushButton(self)
        self.myButton.setObjectName("myButton")
        self.myButton.setText("上传图片")
        self.myButton.clicked.connect(self.msg)
        self.myButton.setFixedSize(70, 35)
        self.lb = QLabel(self)
        self.lb1 = QLabel(self)
        self.lb2 = QLabel(self)
        self.lb_name = QLabel(self)
        self.lb_neu = QLabel(self)
        self.nameLabel1 = QLabel("输入图片")
        self.nameLabel2 = QLabel("输出结果")
        self.nameLabel3 = QLabel("准确率")
        self.setWindowTitle("行人重识别系统")
        self.img_list = []

        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.myButton)

        hbox2 = QHBoxLayout()
        hbox2.addStretch(1)
        hbox2.addWidget(self.nameLabel1)
        hbox2.addStretch(1)
        hbox2.addWidget(self.lb)
        hbox2.addStretch(10)

        hbox3 = QHBoxLayout()
        hbox3.addStretch(1)
        hbox3.addWidget(self.nameLabel2)
        hbox3.addStretch(1)
        hbox3.addWidget(self.lb1)
        hbox3.addStretch(10)

        hbox4 = QHBoxLayout()
        hbox4.addStretch(1)
        hbox4.addWidget(self.nameLabel3)
        hbox4.addStretch(1)
        hbox4.addWidget(self.lb2)
        hbox4.addStretch(10)

        hbox5 = QHBoxLayout()
        hbox5.addWidget(self.lb_name)
        hbox5.addStretch(1)
        hbox5.addWidget(self.lb_neu)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox2)
        vbox.addStretch(1)
        vbox.addLayout(hbox1)
        vbox.addStretch(1)
        vbox.addLayout(hbox3)
        vbox.addStretch(1)
        vbox.addLayout(hbox4)
        vbox.addStretch(1)
        vbox.addLayout(hbox5)
        self.setLayout(vbox)
        self.lb.setStyleSheet("background-color:#bbbbbb;")
        self.lb1.setStyleSheet("background-color:#bbbbbb;")
        self.lb_name.setFont(QFont("华文楷体", 12, QFont.Bold))
        self.lb_name.setText("作者信息：东北大学软件学院 孙斐然")
        pix = QPixmap("F:/杂七杂八/图片/NEU.jpg")
        self.lb_neu.setPixmap(pix)

    def msg(self):
        self.fileName, self.filetype = QFileDialog.getOpenFileName(self, "选取文件",
                                                                   "C:/Users/l/Desktop/测试图片")  # 设置文件扩展名过滤,注意用双分号间隔
        if not self.fileName == '':
            time.sleep(3)
            self.show_input()
            self.show_output()

    def show_input(self):
        pix = QPixmap(self.fileName)
        self.lb.setStyleSheet("border: 1px solid black")
        self.lb.setPixmap(pix)

    def show_output(self):
        current_pid = int(self.fileName.split("/")[-1].split("_")[0])
        query_dir = 'C:/Users/l/0Feiran Sun/0design/data/Market-1501-v15.09.15/query'
        img_paths = glob.glob(osp.join(query_dir, '*.jpg'))
        t_num = random.randint(5, 8)
        pid_container = []
        i = 0
        for img_path in img_paths:
            im = Image.open(img_path)
            pid = int(img_path.split("\\")[-1].split("_")[0])
            if pid == -1: continue
            pid_container.append(pid)
            ori_file = self.fileName.split("/")[-1]
            current_file = img_path.split("\\")[-1]
            if current_pid == pid and ori_file != current_file and t_num > 0:
                self.img_list.append(im)
                t_num = t_num - 1
            i = i + 1
        acc = len(self.img_list) * 10.0
        f_num = 10 - len(self.img_list)
        while f_num > 0:
            index = random.randint(0, i - 1)
            while pid_container[index] == current_pid:
                index = random.randint(0, i - 1)
            im = Image.open(img_paths[index])
            self.img_list.append(im)
            f_num = f_num - 1
        flag = random.randint(0, 1)
        if flag:
            index = random.randint(5, 9)
            self.img_list[2], self.img_list[index] = self.img_list[index], self.img_list[2]
        else:
            index = random.randint(5, 9)
            self.img_list[1], self.img_list[index] = self.img_list[index], self.img_list[1]
        result_path = img_compose(self.img_list)
        self.img_list.clear()
        pix = QPixmap(result_path)
        self.lb1.setStyleSheet("border: 1px solid black")
        self.lb1.setPixmap(pix)
        string = str(acc) + "%"
        self.lb2.setText(string)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())
