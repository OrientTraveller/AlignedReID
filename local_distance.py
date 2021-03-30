import torch

"""
本文件用于定义局部对齐最小距离算法
"""
def batch_euclidean_dist(x, y):
    """
    计算局部特征的欧氏距离
    输入x(N, m, d)，y(N, n, d)
    输出dist(N, m, n)
    其中N为batch_size，m为x的local part，n为y的local part
    """
    assert len(x.size()) == 3#需要判断x是否是三维
    assert len(y.size()) == 3#需要判断y是否是三维
    assert x.size(0) == y.size(0)#需要判断x的第一维的数值是否等于y第一维的数值
    assert x.size(-1) == y.size(-1)#需要判断x的第二维的数值是否等于y第二维的数值
    N, m, d = x.size()
    N, n, d = y.size()
    #经过计算后xx与yy维度都是维度(N, m, n)
    xx = torch.pow(x, 2).sum(-1, keepdim=True).expand(N, m, n)
    #先扩展维度在交换维度的原因时，如果n大于m的话无法使用.expand(N, m, n)函数，因为原始张量第二维数值为n大于目标数值n，所以只能先扩展第三维然后再交换
    yy = torch.pow(y, 2).sum(-1, keepdim=True).expand(N, n, m).permute(0, 2, 1)
    dist = xx + yy
    #计算x^2+y^2-2xy
    dist.baddbmm_(1, -2, x, y.permute(0, 2, 1))#进行三维矩阵相乘，x为N, m, d，y为N, n, d交换维度后N, d, n，最终结果维度N,m,d
    dist = dist.clamp(min=1e-12).sqrt() #维度N,m,d
    return dist

def shortest_dist(dist_mat):
    """
    根据距离矩阵计算局部对齐最小距离
    
    设图片A有局部特征8段，图片B有局部特征6段，则设AB距离矩阵大小为8×6，则dist(3,4)就代表图片A的前4段局部特征与图片B的前5段局部特征的距离。
    由此我们可以知道dist(7,5)就是图片A的前8段局部特征与图片B的前6段局部特征的距离，即图片A与图片B的距离。
    
    在计算最小距离时同时具有局部对齐的作用
    
    输入dist_mat(m, n, N)
    输出dist(N)
    其中N为batch_size，m为x的local part，n为y的local part
    """
    m, n = dist_mat.size()[:2]#获取输入距离矩阵前两维
    dist = [[0 for _ in range(n)] for _ in range(m)]#初始化距离矩阵，类型list，元素也为list
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):#初始化边界
                dist[i][j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):#当i为0时，最小距离只有一种，该种情况属于距离矩阵边界
                dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):#当j为0时，最小距离只有一种，该种情况属于距离矩阵边界
                dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
            else:#在位于距离矩阵内部时，可以选择从上方或从左侧的距离，所以选取其中更小的距离
                dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
    dist = dist[-1][-1]#最后返回距离矩阵右下角元素，即为两局部特征张量的局部对齐最小距离
    return dist

def batch_local_dist(x, y):
    """
    根据局部特征计算最小距离
    输入x(N, m, d)，y(N, n, d)
    输出dist(n)
    """
    assert len(x.size()) == 3
    assert len(y.size()) == 3
    assert x.size(0) == y.size(0)
    assert x.size(-1) == y.size(-1)
    #维度(N, m, n)
    dist_mat = batch_euclidean_dist(x, y)
    #归一化维度(N, m, n)
    #dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
    #输入维度(m, N, n)，输出维度(n)
    dist = shortest_dist(dist_mat.permute(1, 2, 0))
    return dist

if __name__=='__main__':
    x=torch.randn(32,64,64)
    y=torch.randn(32,32,64)
    local_dist=batch_local_dist(x,y)
    print(local_dist)