import torch.nn as nn

"""
用于将全局特征向量水平展开，便于构造局部特征向量
"""


class HorizontalMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        input_size = x.size()
        return nn.functional.max_pool2d(input=x, kernel_size=(1, input_size[3]))


if __name__ == '__main__':
    import torch

    x = torch.rand(32, 2048, 8, 4)
    hmp = HorizontalMaxPool2d()
    print(hmp(x).size())
