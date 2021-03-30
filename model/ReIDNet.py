import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from .HorizontalMaxPool2d import HorizontalMaxPool2d
from IPython import embed 
"""
该文件指定网络的结构
根据“是否计算局部特征、损失函数类型”决定网络返回哪些计算结果
"""
class ReIDNet(nn.Module):
    """
    自定义的网络需要继承torch.nn当中的Module类
    Module.py类是所有神经网络的基础类，自定义的网络类需要继承此类，同时需要重写其中的两个方法
    __init__()，用于初始化网络
    forward()，用于网络的计算前向传播结果（反向传播不需要自己编写）
    """
    def __init__(self, num_classes, loss={'softmax, metric'}, aligned=False,**kwargs):
        #第一个参数num_classes指定网络训练时输出的分类结果有几个类
        #第二个参数loss指定训练网络使用的损失函数
        #第三个参数aligned指定该网络是否使用AlignedReID的局部特征分支
        super().__init__()
        orinet=torchvision.models.alexnet(pretrained=True)#预训练参数指定为True，预训练的数据集是ImageNet数据集
        self.loss=loss
        #进行基础网络修改
        """
        使用list(yournet.children())将网络按照层次分开
        list前面加星号将列表内元素每一个都作为nn.Sequential单独参数形成一个新的网络层序列
        ！！注意nn.Sequential函数不能是list，只能是网络层！！
        """
        self.basenet=nn.Sequential(*list(orinet.children())[0:1])
        #将特征向量转化到目标规模
        self.linear=nn.Linear(256, 2048)
        #分类器
        self.classifier=nn.Linear(2048, num_classes)
        self.aligned=aligned
        #如果使用局部特征分支，则需要指定一些用到的网络层与函数
        if self.aligned:
            self.horizon_pool=HorizontalMaxPool2d()#自定义水平最大池化层，用于展开特征向量
            self.bn=nn.BatchNorm2d(256)#用于归一化
            self.relu=nn.ReLU(inplace=True)
            self.conv1=nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0,bias=True)#计算局部特征使用的卷积层
    def forward(self, x):
        x=self.basenet(x)
        cnn_result=F.avg_pool2d(x, x.size()[2:]).view(x.size(0),-1)
        f=self.linear(cnn_result)
        #归一化
        f=1.*f / (torch.norm(f,2,dim=-1, keepdim=True).expand_as(f)+1e-12)
        #如果是用于测试，只返回feature张量
        if not self.training:
            return f
        #如果使用Alinged分支，则加以计算local feature；如果不使用Alinged分支，则根据度量学习与表征学习分类
        if self.aligned:
            lf=self.bn(x)
            lf=self.relu(lf)
            lf=self.horizon_pool(lf)
            lf=self.conv1(lf)
            y=self.classifier(f)
            return y,f,lf
        #根据损失函数决定返回哪些计算结果
        else:
            if self.loss=={'softmax'}:#表征学习
                y=self.classifier(f)
                return y
            elif self.loss=={'metric'}:#度量学习
                return f
            elif self.loss=={'softmax, metric'}:#表征学习度量学习结合
                y=self.classifier(f)
                return y,f
            else:
                print('loss setting error')

if __name__=='__main__':
    model=ReIDNet(num_classes=751,aligned=True)
    imgs=torch.rand(32,3,128,64)
    y,f,lf=model(imgs)
    print(y.size())
    print(f.size()) 
    print(lf.size())