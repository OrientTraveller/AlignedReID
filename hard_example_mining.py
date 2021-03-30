import torch

"""
本文件用于定义难样本挖掘算法
输入：
1.距离矩阵dist_mat，维度(batch_size,batch_size)
2.本批次特征向量对应的行人ID，维度(batch_size)
3.是否返回最小相似度正样本与最大相似度负样本所对应的距离矩阵的序号return_indexs，默认为False

输出：
1.正样本区最小相似度张量dist_ap，维度(batch_size)
2.负样本区最大相似度张量dist_an，维度(batch_size)
3.正样本区最小相似度样本对应的距离矩阵下标p_indexs，维度(batch_size)
4.负样本区最大相似度样本对应的距离矩阵下标,n_indexs，维度(batch_size)
"""


def hard_example_mining(dist_mat, labels, return_inds=False):
    # 先判断距离矩阵是不是二维，若不是二维则报错
    assert len(dist_mat.size()) == 2
    # 判断距离矩阵是否是方阵，若不是方阵则报错
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)  # 获取方阵长度
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())  # 正样本区掩码，负样本区元素为0
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())  # 负样本区掩码，正样本区元素为0

    """
    .contiguous()用于将正样本区的距离拉成一维连续向量
    .view(N,-1)用于按照N为行形成矩阵
    torch.max函数不仅返回每一列中最大值的那个元素，并且返回最大值在这一行中对应索引
    """
    # 计算最小相似度（最大距离）正样本距离与最小相似度所对应正样本在某一行中的序号（序号范围0~n-1）
    dist_ap, relative_p_indexs = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)

    # 计算最大相似度（最小距离）负样本距离与最大相似度所对应负样本在某一行中的序号（序号范围0~n-1）
    dist_an, relative_n_indexs = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)

    # 上面计算得到的dist_ap与dist_an维度为(batch_size,1)需要将最后一维进行压缩
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    # 根据计算得到的序号计算最小相似度正样本与最大相似度负样本在距离矩阵中的序号
    if return_inds:
        indexs = (labels.new().resize_as_(labels).copy_(torch.arange(0, N).long()).unsqueeze(0).expand(N, N))
        """
        gather函数的用法torch.gather(input, dim, index, out=None) 
        就是从index中找到某值，作为input的某一维度的索引，取出的input的值作为output的某一元素
        核心思想：
        out[i][j][k] = input[index[i][j][k]] [j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]   # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]   # if dim == 2
        注，output的维度同index一样
        
        对于下式就是p_indexs[i][0]=indexs[is_neg].contiguous().view(N, -1)[i][relative_n_indexs.data[i][0]]
        因为index即relative_n_indexs.data最后一维值为1，所以只能等于0
        """
        # 计算最小相似度正样本与最大相似度负样本在距离矩阵中的序号，结果维度(batch_size,1)
        p_indexs = torch.gather(indexs[is_pos].contiguous().view(N, -1), 1, relative_p_indexs.data)
        n_indexs = torch.gather(indexs[is_neg].contiguous().view(N, -1), 1, relative_n_indexs.data)

        # 将结果最后一维压缩
        p_indexs = p_indexs.squeeze(1)
        n_indexs = n_indexs.squeeze(1)

        return dist_ap, dist_an, p_indexs, n_indexs
    return dist_ap, dist_an
