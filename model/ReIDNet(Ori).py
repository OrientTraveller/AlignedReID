import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from IPython import embed

class ReIDNet(nn.Module):
    def __init__(self, num_classes, loss={'softmax, metric'}, **kwargs):
        super().__init__()
        orinet=torchvision.models.alexnet(pretrained=True)
        self.loss=loss
        self.basenet=nn.Sequential(*list(orinet.children())[0:1])
        self.linear=nn.Linear(256, 2048)
        self.classifier=nn.Linear(2048, num_classes)
    def forward(self, x):
        x=self.basenet(x)
        cnn_result=F.avg_pool2d(x, x.size()[2:]).view(x.size(0),-1)
        f=self.linear(cnn_result)
        #归一化
        f=1.*f / (torch.norm(f,2,dim=-1, keepdim=True).expand_as(f)+1e-12)
        #以下if/else用于控制网络训练与测试所使用的网络结构
        if not self.training:
            return f
        if self.loss=={'softmax'}:
            y=self.classifier(f)
            return y
        elif self.loss=={'metric'}:
            return f
        elif self.loss=={'softmax, metric'}:
            y=self.classifier(f)
            return y,f
        else:
            print('loss setting error')

if __name__=='__main__':
    model=ReIDNet(num_classes=751)
    imgs=torch.rand(32,3,128,64)
    f=model(imgs)
    print(imgs.size())
    print(f.size())
    embed()




