import os,sys,time,datetime
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import transform as T
from IPython import embed
from torch.utils.data import DataLoader

from model.ReIDNet import ReIDNet
from dataset_manager import Market1501
from dataset_loader import ImageDataset
from distance import low_memory_local_dist #用于计算局部特征距离矩阵，使用局部对齐最小距离算法
from eval_metrics import evaluate                 #用于评估算法性能（CMC曲线Rank-N与mAP）

"""
该文件用于编写模型测试函数，进行模型性能指标的考量
"""
#设定输入参数
width=64                    #图片宽度
height=128                 #图片高度
batch_size=32           #训练批量
aligned=True              #在计算局部特征距离矩阵时是否使用局部对齐最小距离算法

"""
test函数用于测试模型的性能，具体的性能指标为CMC与mAP
输入：
1.需要进行测试的模型model
2.query集数据吞吐器queryloader
3.gallery集数据吞吐器galleryloader
4.需要计算的rank准确率ranks，类型列表
"""
def test(model, queryloader, galleryloader,ranks=[1, 5, 10, 20]):
    #设定模型为测试模式
    model.eval()
    
    #with torch.no_grad():使with中的语句不会计算梯度，计算过程不会在反向传播中被记录
    with torch.no_grad():
        
        #以下都是列表。qf存储全局特征；q_pids存储行人ID；q_camids存储摄像机ID；lqf存储局部特征
        qf, q_pids, q_camids, lqf = [], [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            features, local_features = model(imgs)#计算全局特征与局部特征
            qf.append(features)
            lqf.append(local_features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)#将全局特征按列形成一个张量
        lqf = torch.cat(lqf,0)#将局部特征按列形成一个张量
        q_pids = np.asarray(q_pids)#将行人ID列表转成np数组
        q_camids = np.asarray(q_camids)#将摄像机ID列表转成np数组
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        #以下都是列表。gf存储全局特征；g_pids存储行人ID；g_camids存储摄像机ID；lgf存储局部特征
        gf, g_pids, g_camids, lgf = [], [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            features, local_features = model(imgs)#计算全局特征与局部特征
            gf.append(features)
            lgf.append(local_features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)#将全局特征按列形成一个张量
        lgf = torch.cat(lgf,0)#将局部特征按列形成一个张量
        g_pids = np.asarray(g_pids)#将行人ID列表转成np数组
        g_camids = np.asarray(g_camids)#将摄像机ID列表转成np数组
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    #特征标准化
    qf = 1. * qf / (torch.norm(qf, 2, dim = -1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim = -1, keepdim=True).expand_as(gf) + 1e-12)
    
    """
    计算query特征与gallery特征的距离矩阵
    全局特征计算欧氏距离，矩阵A,B欧氏距离等于√(A^2 + (B^T)^2 - 2A(B^T))
    局部特征使用局部对齐最小距离算法计算距离
    """
    m, n = qf.size(0), gf.size(0)
    #计算A^2 + (B^T)^2
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)+torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    #计算A^2 + (B^T)^2 - 2A(B^T)
    distmat.addmm_(1, -2, qf, gf.t())
    #计算√(A^2 + (B^T)^2 - 2A(B^T))
    distance=distance.clamp(min=1e-12).sqrt()
    #转换成numpy数组
    global_distmat = distmat.numpy()
    
    lqf = lqf.permute(0,2,1)
    lgf = lgf.permute(0,2,1)
    #计算局部特征距离矩阵
    local_distmat = low_memory_local_dist(lqf.numpy(),lgf.numpy(),aligned= aligned)
    
    #得到包含全局特征与局部特征的总距离矩阵
    print("Using global and local branches")
    distmat = local_distmat+global_distmat
    
    print("Computing CMC and mAP")
    #使用评估函数计算cmc曲线中的Rank-N与mAP平均准确率均值
    cmc, mAP = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids)
    #打印相关性能测试结果
    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("-------------------")
    
    return 0

if __name__=='__main__':
    #使用局部对齐模型
    model=ReIDNet(num_classes=751,loss={'softmax, metric'},aligned=True)
    #加载局部对齐模型最优参数
    model.load_state_dict(torch.load('./model/param/aligned_trihard_net_params_best.pth'))
    #指定数据集
    dataset=Market1501()
    #query数据与gallery数据处理器
    transform=T.Compose([
        T.Resize((height,width)),#尺度统一
        T.ToTensor(),#图片转张量
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),#归一化，参数固定
    ]
    )
    #query集吞吐器
    query_data_loader=DataLoader(
        ImageDataset(dataset.query, transform=transform),#自定义的数据集，指定使用数据处理器
        batch_size=batch_size,#一个批次的大小（一个批次有多少个图片张量）
        drop_last=True,#丢弃最后无法称为一整个批次的数据
    )
    #gallery集吞吐器
    gallery_data_loader=DataLoader(
        ImageDataset(dataset.gallery, transform=transform),#自定义的数据集，指定使用数据处理器
        batch_size=batch_size,#一个批次的大小（一个批次有多少个图片张量）
        drop_last=True,#丢弃最后无法称为一整个批次的数据
    )
    #调用test函数进行算法性能评估
    test(model,query_data_loader,gallery_data_loader)