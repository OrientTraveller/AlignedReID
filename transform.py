import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止服务挂掉

"""
模型读取数据（无论是训练还是测试）时，需要对数据进行必要的处理，如尺度统一、水平变换（训练时需要，测试时不需要）、将图片转为张量、归一化等
该文件用于对数据集图片进行尺寸标准化与随机裁剪
"""


class Random2DTransform(object):
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):  # 目标高度、目标宽度、概率p（用于是否进行随机裁剪）、插值方法（默认线性插值）
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        # 若小于目标概率，则直接将图片尺寸标准化
        if random.random() < self.p:
            img = img.resize((self.width, self.height), self.interpolation)
        # 若大于目标概率，则先根据插值方法扩大图片，并进行随机裁剪
        else:
            new_width = int(round(self.width * 1.125))
            new_height = int(round(self.height * 1.125))
            # 先将图片扩大到目标长宽的1.125倍，然后再随机裁剪
            resize_img = img.resize((new_width, new_height), self.interpolation)
            x_maxrange = new_width - self.width
            y_maxrange = new_height - self.height
            # 计算随机裁剪XY轴起点
            x_start = int(round(random.uniform(0, x_maxrange)))
            y_start = int(round(random.uniform(0, y_maxrange)))
            # 进行裁剪
            img = resize_img.crop((x_start, y_start, x_start + self.width, y_start + self.height))
        return img


if __name__ == '__main__':
    from dataset_manager import Market1501
    from dataset_loader import ImageDataset

    dataset = Market1501()
    train_loader = ImageDataset(dataset.train)
    plt.figure()
    j = 1
    # 从训练集中获取前两张图片进行处理，并使用matplot显示图片
    for batch_id, (img, pid, cid) in enumerate(train_loader):
        if (batch_id < 2):
            transform = Random2DTransform(64, 64, 0.5)
            img_t = transform(img)
            img_t = np.array(img_t)
            plt.subplot(1, 2, j)
            plt.imshow(img)  # 显示图片
            plt.savefig()
            j = j + 1
            plt.subplot(1, 2, j)
            plt.imshow(img_t)  # 显示图片
            plt.show()
            j = 1
