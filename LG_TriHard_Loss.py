import torch
from torch import nn
from local_distance import batch_local_dist
from hard_example_mining import hard_example_mining

"""
本文件用于自定义计算全局特征与局部特征的难样本挖掘三元组损失
需要调用先前编写的难样本挖掘算法与局部对齐最小距离算法
"""


class AlignedTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        # margin就是三元组损失中的边界α
        self.margin = margin
        # 计算三元组损失使用的函数
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, local_features, targets):
        """
        输入：
        1.全局特征张量inputs
        2.局部特征张量local_features
        3.真实行人IDtargets
        
        输出：
        1.全局特征损失global_loss
        2.局部特征损失,local_loss
        """
        # 获取批量
        n = inputs.size(0)

        # 将局部特征张量进行维度压缩
        local_features = local_features.squeeze()

        """
        计算图片之间的欧氏距离
        矩阵A,B欧氏距离等于√(A^2 + (B^T)^2 - 2A(B^T))
        """
        # 计算A^2
        distance = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # 计算A^2 + (B^T)^2
        distance = distance + distance.t()
        # 计算A^2 + (B^T)^2 - 2A(B^T)
        distance.addmm(1, -2, inputs, inputs.t())
        # 计算√(A^2 + (B^T)^2 - 2A(B^T))
        distance = distance.clamp(min=1e-12).sqrt()  # 该distance矩阵为对称矩阵

        # 获取正负样本对距离，使用难样本挖掘
        dist_ap, dist_an, p_inds, n_inds = hard_example_mining(distance, targets, return_inds=True)
        p_inds, n_inds = p_inds.long(), n_inds.long()

        # 根据难样本挖掘计算得到最小相似度正样本与最大相似度负样本索引，提取对应难样本的局部特征
        p_local_features = local_features[p_inds]
        n_local_features = local_features[n_inds]

        # 对难样本局部特征使用局部对齐最小距离算法计算样本对距离
        local_dist_ap = batch_local_dist(local_features, p_local_features)
        local_dist_an = batch_local_dist(local_features, n_local_features)

        # y指明ranking_loss前一个参数大于后一个参数
        y = torch.ones_like(dist_an)
        # 全局特征损失
        global_loss = self.ranking_loss(dist_an, dist_ap, y)
        # 局部特征损失
        local_loss = self.ranking_loss(local_dist_an, local_dist_ap, y)

        return global_loss, local_loss


if __name__ == '__main__':
    target = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8]
    target = torch.Tensor(target)
    features = torch.rand(32, 2048)
    local_features = torch.randn(32, 128, 3)
    a = AlignedTripletLoss()
    g_loss, l_loss = a.forward(features, local_features, target)
    print(g_loss)
    print(l_loss)
