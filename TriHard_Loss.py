import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from IPython import embed

"""
本文件用于自定义难样本挖掘三元组损失，定义难样本挖掘三元组损失计算过程。
"""
class TripletLoss(nn.Module):
    """
    继承nn.Module，重写__init__方法与forward方
    """
    def __init__(self, margin=0.3):
        super().__init__()
        #margin就是三元组损失中的边界α
        self.margin=margin
        #计算三元组损失使用的函数
        self.ranking_loss=nn.MarginRankingLoss(margin=margin)
        
    def forward(self, inputs, targets):
        """
        输入：
        1.全局特征张量inputs
        2.真实行人IDtargets
        
        输出：
        1.特征损失loss
        """
        n=inputs.size(0)
        """
        计算图片之间的欧氏距离
        矩阵A,B欧氏距离等于√(A^2 + (B^T)^2 - 2A(B^T))
        """
        #计算A^2
        distance=torch.pow(inputs,2).sum(dim=1, keepdim=True).expand(n,n)
        #计算A^2 + (B^T)^2
        distance=distance+distance.t()
        #计算A^2 + (B^T)^2 - 2A(B^T)
        distance.addmm(1,-2,inputs,inputs.t())
        #计算√(A^2 + (B^T)^2 - 2A(B^T))
        distance=distance.clamp(min=1e-12).sqrt()#该distance矩阵为对称矩阵
        
        #获取对角线
        mask=targets.expand(n,n)==targets.expand(n,n).t()#mask矩阵用于区分红绿色区域，即正样本区与负样本区，便于进行损失计算。
        
        #list类型
        distance_ap,distance_an=[],[]
        
        for i in range(n):
            distance_ap.append(distance[i][mask[i]].max().unsqueeze(0))#distance[i][mask[i]]使distance保留正样本区
            distance_an.append(distance[i][mask[i]==0].min().unsqueeze(0))#distance[i][mask[i]==0]使distance保留负样本区
        
        #经过for循环后，正样本最大距离与负样本最小距离都存储在list当中，需要将list元素连接成一个torch张量
        distance_ap=torch.cat(distance_ap)
        distance_an=torch.cat(distance_an)
        #y指明ranking_loss前一个参数大于后一个参数
        y=torch.ones_like(distance_an)
        loss=self.ranking_loss(distance_an, distance_ap, y)
        
        return loss

if __name__=='__main__':
    target=[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8]
    target=torch.Tensor(target)
    features=torch.rand(32,2048)
    a=TripletLoss()
    loss=a.forward(features,target)
    print(loss)