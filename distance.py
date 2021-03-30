import numpy as np

"""
本文件用于计算局部特征距离矩阵，使用局部对齐最小距离算法
"""


def low_memory_local_dist(x, y, aligned=True):
    """
    主函数low_memory_local_dist，将输入特征张量x,y进行分块（分而治之思想），防止大量数据溢入内存，并调用low_memory_matrix_op计算局部特征距离矩阵
    
    输入：
    1.query集或gallery集局部特征张量x
    2.query集或gallery集局部向量特征张量y
    3.是否使用局部对齐最小距离算法bool类型变量aligned默认为True
    
    输出：
    1.局部特征距离矩阵dismat
    """
    print('Computing local distance...')
    x_num_splits = int(len(x) / 200) + 1  # 计算张量x的分块数量
    y_num_splits = int(len(y) / 200) + 1  # 计算张量y的分块数量
    # 若下函数运行过程爆出内存溢出，则增加x_num_splits与y_num_splits
    dismat = low_memory_matrix_op(local_dist, x, y, 0, 0, x_num_splits, y_num_splits,
                                  aligned=aligned)  # 默认在0轴上分块，按行分块即若干行一块
    return dismat


def low_memory_matrix_op(func, x, y, x_split_axis, y_split_axis, x_num_splits, y_num_splits, aligned=True):
    """
    函数low_memory_matrix_op根据x_split_axis, y_split_axis, x_num_splits, y_num_splits四个参数对张量x,y进行分块，
    然后对每一个分块使用指定的函数func计算距离矩阵，然后将距离矩阵进行横向、纵向拼接，得到总的距离矩阵并返回
    
    输入：
    1.形成距离矩阵的函数func
    2.query集或gallery集局部特征张量x
    3.query集或gallery集局部向量特征张量y
    4.张量x的分块的轴x_split_axis
    5.张量y的分块的轴y_split_axis
    6.张量x的分块数量x_num_splits
    7.张量y的分块数量y_num_splits
    8.是否使用局部对齐最小距离算法bool类型变量aligned，默认为True
    
    输出：
    1.距离矩阵mat
    """
    # 存储结果的矩阵mat，因为是按行分块，所以第一维为空，第二维为x的分块数量
    mat = [[] for _ in range(x_num_splits)]

    # 对于分块的每一部分
    for i, part_x in enumerate(np.array_split(x, x_num_splits, axis=x_split_axis)):
        for j, part_y in enumerate(np.array_split(y, y_num_splits, axis=y_split_axis)):
            part_mat = func(part_x, part_y, aligned)  # 调用func函数计算该分块的距离矩阵
            mat[i].append(part_mat)  # 将分块计算所得距离矩阵加入结果中
        mat[i] = np.concatenate(mat[i], axis=1)  # 在1轴上进行数组水平拼接（横向拼接）

    mat = np.concatenate(mat, axis=0)  # 在0轴上进行数组垂直拼接（纵向拼接）
    # 返回计算完毕的距离矩阵
    return mat


def local_dist(x, y, aligned):
    """
    函数local_dist根据张量x,y的维度分别使用不同的函数进行距离矩阵计算
    
    输入：
    1.query集或gallery集局部特征张量x
    2.query集或gallery集局部向量特征张量y
    3.是否使用局部对齐最小距离算法bool类型变量aligned
    
    输出：
    1.距离矩阵计算函数返回值
    """
    # 若张量x,y都是二维
    if (x.ndim == 2) and (y.ndim == 2):
        return meta_local_dist(x, y, aligned)
    # 若张量x,y都是三维
    elif (x.ndim == 3) and (y.ndim == 3):
        return parallel_local_dist(x, y, aligned)
    # 否则报错，输入规模不支持
    else:
        raise NotImplementedError('输入规模不支持距离矩阵计算.')


def meta_local_dist(x, y, aligned):
    """
    函数meta_local_dist根据张量x,y先计算原始距离矩阵然后对距离矩阵进行处理（可以选用局部对齐最小距离算法也可以不用）得到结果距离矩阵
    
    输入：
    1.query集或gallery集局部特征张量x
    2.query集或gallery集局部向量特征张量y
    3.是否使用局部对齐最小距离算法bool类型变量aligned
    
    输出：
    1.距离矩阵
    """
    # 先计算欧氏距离的距离矩阵
    eu_dist = compute_dist(x, y, 'euclidean')
    # 对距离矩阵进行归一化
    dist_mat = (np.exp(eu_dist) - 1.) / (np.exp(eu_dist) + 1.)
    # 如果声明使用局部对齐最小距离算法，则使用shortest_dist函数
    if aligned:
        dist = shortest_dist(dist_mat[np.newaxis])[0]
    # 如果声明不使用局部对齐最小距离算法，则使用unaligned_dist函数
    else:
        dist = unaligned_dist(dist_mat[np.newaxis])[0]
    return dist


def parallel_local_dist(x, y, aligned):
    """
    函数parallel_local_dist根据张量x,y先计算原始距离矩阵然后对距离矩阵进行处理（可以选用局部对齐最小距离算法也可以不用）得到结果距离矩阵
    
    输入：
    1.query集或gallery集局部特征张量x
    2.query集或gallery集局部向量特征张量y
    3.是否使用局部对齐最小距离算法bool类型变量aligned
    
    输出：
    1.距离矩阵
    """
    # 获取张量x,y的规模
    M, m, d = x.shape
    N, n, d = y.shape

    # 改变张量x,y的规模，将其变成2维
    x = x.reshape([M * m, d])
    y = y.reshape([N * n, d])

    # 距离矩阵[M × m,N × n]
    # 先计算欧氏距离的距离矩阵
    dist_mat = compute_dist(x, y, type='euclidean')
    # 对距离矩阵进行归一化
    dist_mat = (np.exp(dist_mat) - 1.) / (np.exp(dist_mat) + 1.)
    # 将距离矩阵规模从[M × m,N × n]变为[M, m, N, n]再变为[m, n, M, N]
    dist_mat = dist_mat.reshape([M, m, N, n]).transpose([1, 3, 0, 2])
    # 如果声明使用局部对齐最小距离算法，则使用shortest_dist函数
    if aligned:
        dist_mat = shortest_dist(dist_mat)
    # 如果声明不使用局部对齐最小距离算法，则使用unaligned_dist函数
    else:
        dist_mat = unaligned_dist(dist_mat)
    return dist_mat


def compute_dist(array1, array2, type='euclidean'):
    """
    函数compute_dist用于计算两矩阵的欧氏距离或余弦距离
    
    输入：
    1.矩阵1array1，规模[m1, n]
    2.矩阵2array2，规模[m2, n]
    3.距离的类型type，可选'cosine'或'euclidean'
    
    输出：
    1.距离矩阵dist，规模[m1, m2]
  """
    assert type in ['cosine', 'euclidean']
    # 如果指定type == 'cosine'，则计算余弦距离
    if type == 'cosine':
        array1 = normalize(array1, axis=1)  # 分步计算余弦距离
        array2 = normalize(array2, axis=1)  # 分步计算余弦距离
        dist = np.matmul(array1, array2.T)  # 矩阵相乘
        return dist
    # 如果指定type == 'euclidean'，则计算欧氏距离
    else:
        # 矩阵A,B欧氏距离等于√(A^2 + (B^T)^2 - 2A(B^T))
        # 计算A^2
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # 计算(B^T)^2
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        # 计算A^2 + (B^T)^2- 2A(B^T)
        squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
        # 将矩阵中负数置为0
        squared_dist[squared_dist < 0] = 0
        # 计算√(A^2 + (B^T)^2 - 2A(B^T))
        dist = np.sqrt(squared_dist)
        return dist


def normalize(nparray, order=2, axis=0):
    """
    函数normalize用于辅助计算矩阵余弦距离
    """
    # 求矩阵的范数，默认order=2, axis=0求整个矩阵的2范数
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def shortest_dist(dist_mat):
    """
    函数shortest_dist使用局部对齐最小距离算法计算结果距离矩阵
    
    输入：
    1.距离矩阵dist_mat，该距离要么是欧氏距离，要么是余弦距离
      dist_mat有如下可能的维度：
      1) [m, n]
      2) [m, n, N], N是batch_size
      3) [m, n, *], *代表增加的维度
    
    输出：
    1.距离矩阵dist
      对应上面的三种情况：
      1) 标量
      2) 矩阵，规模为[N]
      3) 矩阵，规模为[*]
    """
    # 获取矩阵前两维的数值
    m, n = dist_mat.shape[:2]
    # 局部对齐最小距离算法
    dist = np.zeros_like(dist_mat)
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):  # 初始化边界
                dist[i, j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):  # 当i为0时，最小距离只有一种，该种情况属于距离矩阵边界
                dist[i, j] = dist[i, j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):  # 当j为0时，最小距离只有一种，该种情况属于距离矩阵边界
                dist[i, j] = dist[i - 1, j] + dist_mat[i, j]
            else:  # 在位于距离矩阵内部时，可以选择从上方或从左侧的距离，所以选取其中更小的距离
                dist[i, j] = np.min(np.stack([dist[i - 1, j], dist[i, j - 1]], axis=0), axis=0) + dist_mat[i, j]
    dist = dist[-1, -1].copy()  # 最后返回距离矩阵右下角元素，即为两局部特征张量的局部对齐最小距离
    return dist


def unaligned_dist(dist_mat):
    """
    函数unaligned_dist直接根据原始距离矩阵返回结果距离矩阵
    
    输入：
    1.距离矩阵dist_mat，该距离要么是欧氏距离，要么是余弦距离
      dist_mat有如下可能的维度：
      1) [m, n]
      2) [m, n, N], N是batch_size
      3) [m, n, *], *代表增加的维度
    
    输出：
    1.距离矩阵dist
      对应上面的三种情况：
      1) 标量
      2) 矩阵，规模为[N]
      3) 矩阵，规模为[*]
    """
    # 获取矩阵第一维的数值
    m = dist_mat.shape[0]
    dist = np.zeros_like(dist_mat[0])
    for i in range(m):
        dist[i] = dist_mat[i][i]  # 不做任何处理，直接返回计算得到的原始距离
    dist = np.sum(dist, axis=0).copy()  # 只是将局部特征的距离进行线性加和，没有使用局部对齐最小距离算法
    return dist
