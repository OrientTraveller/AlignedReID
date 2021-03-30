import os
import os.path as osp
import numpy as np
import glob
import re
from IPython import embed

"""
Market1501类用于
1.指定数据集路径
2.处理原始数据集并生成数据索引列表
3.返回子数据集的相关参数（子集行人ID数量，子集图片数量）
"""
class Market1501(object):
    dataset_dir='data/Market-1501-v15.09.15'#指定数据集路径
    
    def __init__(self,root='./',**kwargs):
        self.dataset_dir=osp.join(root,self.dataset_dir)
        self.train_dir=osp.join(self.dataset_dir,'bounding_box_train')#训练集
        self.gallery_dir=osp.join(self.dataset_dir,'bounding_box_test')#测试集
        self.query_dir=osp.join(self.dataset_dir,'query')#查询集
        
        train, num_train_pids, num_train_imgs=self._process_dir(self.train_dir,relabel=True)
        query, num_query_pids, num_query_imgs=self._process_dir(self.query_dir,relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs=self._process_dir(self.gallery_dir,relabel=False)
        
        num_total_pids=num_train_pids+num_query_pids
        num_total_imgs=num_train_imgs+num_query_imgs
        
        print("=> Market1501 loaded")
        print("------------------------------------------------------------------------")
        print("  subset: train  \t| num_id: {:5d}  \t|  num_imgs:{:8d}  ".format(num_train_pids,num_train_imgs))
        print("  subset: query  \t| num_id: {:5d}  \t|  num_imgs:{:8d}  ".format(num_query_pids,num_query_imgs))
        print("  subset: gallery \t| num_id: {:5d}  \t|  num_imgs:{:8d}  ".format(num_gallery_pids,num_gallery_imgs))
        print("------------------------------------------------------------------------")
        print("  total \t\t\t| num_id: {:5d}  \t|  num_imgs:{:8d}  ".format(num_total_pids,num_total_imgs))
        print("------------------------------------------------------------------------")
        
        self.train=train
        self.query=query
        self.gallery=gallery
        self.num_train_pids=num_train_pids
        self.num_query_pids=num_query_pids
        self.num_gallery_pids=num_gallery_pids
        
    def _process_dir(self,dir_path,relabel=False):
        img_paths=glob.glob(osp.join(dir_path,'*.jpg'))
        pid_container=set()
        
        for img_path in img_paths:
            pid=int(img_path.split("\\")[-1].split("_")[0])
            if pid==-1:continue
            pid_container.add(pid)
        
        pid2label={pid:label for label,pid in enumerate(pid_container)}
        
        dataset=[]
        
        for img_path in img_paths:
            str_list=img_path.split("\\")[-1].split("_")
            pid=int(str_list[0])
            cid=int(str_list[1][1:2])
            if pid==-1:continue
            assert 0<=pid <=1501
            assert 1<=cid<=6
            cid+=-1
            if relabel:
                pid=pid2label[pid]
            dataset.append((img_path,pid,cid))
        
        num_pids=len(pid_container)
        num_imgs=len(img_paths)
        #返回一个数据为三元组（图片地址，行人ID，摄像机ID）的索引列表形式的数据集，行人ID数量，图片数量
        return dataset, num_pids, num_imgs    

if __name__=='__main__':
    data=Market1501()