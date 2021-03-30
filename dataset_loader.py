import os
from PIL import Image
import numpy as np
import os.path as osp
import torch
from torch.utils.data import Dataset

"""
ImageDataset类用于
将dataset_manager生成的元素为（图片地址，行人ID，摄像机ID）的索引列表数据集转换为元素为（图片，行人ID，摄像机ID）的图片数据集
即：（图片地址，行人ID，摄像机ID）——>（图片，行人ID，摄像机ID）
"""
def read_image(img_path):
    if not osp.exists(img_path):
        raise IOError("{} does not exist.".format(img_path))
    else:
        img=Image.open(img_path).convert('RGB')
        return img

class ImageDataset(Dataset):
    """
    所有继承torch.utils.data.Dataset类都需要重写其中的__len__与__getitem__方法
    前者用于获取数据集的大小
    后者用于数据集迭代过程中获取其中每一条数据
    """
    def __init__(self,dataset, transform=None):
        self.dataset=dataset
        self.transform=transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img_path, pid, cid=self.dataset[index]
        img=read_image(img_path)
        if self.transform:#如果指定了transform的方法，则使用transform进行数据转换
            img = self.transform(img)
        return img, pid, cid

if __name__=='__main__':
    from dataset_manager import Market1501
    dataset=Market1501()
    train_loader=ImageDataset(dataset.train)
    from IPython import embed
    for batch_id, (img, pid, cid) in enumerate(train_loader):
        print(batch_id,img,pid,cid)