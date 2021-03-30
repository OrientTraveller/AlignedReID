import os
import os.path as osp
import glob
import cv2

"""
dataset_preprocess
使用OpenCV自带的行人检测器对训练图片进行筛选，将那些无法识别出行人的训练图片进行删除
"""

if __name__ == "__main__":
    # 指定数据集路径
    train_dataset_dir = 'data/test'
    # 使用opencv的hog特征进行行人检测
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # 获得数据集下图片地址集合
    img_paths = glob.glob(osp.join(train_dataset_dir, '*.jpg'))
    # 开始进行检测处理
    for img_path in img_paths:
        img = cv2.imread(img_path)  # 读取图片
        # 使用行人检测器检测，第一个结果是检测到的行人的坐标，第二个结果是每个行人的置信值
        results, weights = detector.detectMultiScale(img, padding=(16, 16))
        if len(results) == 0:
            os.remove(img_path)
