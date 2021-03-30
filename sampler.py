import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict
from IPython import embed

"""
本文件用于自定义采样器类RandomIdentitySampler
RandomIdentitySampler根据指定的数据集（索引列表）与采样数量进行采样，最后返回记录采样数据图片序号的列表迭代器
在度量学习中，模型需要通过网络学习出图片间的相似度，相同行人图片相似度高于不同行人。所以需要一个批次中既有相同行人图片也要有不同行人图片，所以需要使用随机采样器进行采样。
"""
class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source=data_source
        self.num_instances=num_instances
        self.index_dic= defaultdict(list)#改字典用于记录行人ID与其数据图片序号的对应关系
        for index, (_,pid,_) in enumerate(data_source):
            self.index_dic[pid].append(index)#将该行人ID对应的数据图片序号加入字典
        self.pids=list(self.index_dic.keys())#获取行人ID列表
        self.num_indentities=len(self.pids)
    def __iter__(self):
        indices=torch.randperm(self.num_indentities)#对行人ID打乱顺序重排
        result=[]#result列表用于存储采样数据图片的序号
        for i in indices:
            #注意类型转换，字典的索引不能是Tensor类型
            t=self.index_dic[int(i)]
            #如果该pid拥有的图片少于num_instances，则可以重复采样
            replace=False if len(t)>=self.num_instances else True
            #采样
            t=np.random.choice(t, size=self.num_instances, replace=replace)
            result.extend(t)
        #！！一定要返回result的迭代器，不能只返回result列表！！
        return iter(result)
    def __len__(self):
        return self.num_instances*self.num_indentities

if __name__=='__main__':
    from dataset_manager import Market1501
    dataset=Market1501()
    print(type(dataset.train))
    sampler=RandomIdentitySampler(dataset.train, num_instances=4)
    b=sampler.__iter__()
    print('采样样本list的长度为{}'.format(sampler.__len__()))