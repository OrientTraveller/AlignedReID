import torch
import torchvision
from torch import nn
from torch.nn import functional as func
from IPython import embed


class ReIDNet(nn.Module):
    def __init__(self, num_classes, loss=None):
        super().__init__()
        if loss is None:
            loss = {'softmax, metric'}
        orinet = torchvision.models.alexnet(pretrained=True)
        self.loss = loss
        self.basenet = nn.Sequential(*list(orinet.children())[0:1])
        self.linear = nn.Linear(256, 2048)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.basenet(x)
        cnn_result = func.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        feature = self.linear(cnn_result)
        # 归一化
        feature = 1. * feature / (torch.norm(feature, 2, dim=-1, keepdim=True).expand_as(feature) + 1e-12)
        # 以下if/else用于控制网络训练与测试所使用的网络结构
        if not self.training:
            return feature
        if self.loss == {'softmax'}:
            classification_vector = self.classifier(feature)
            return classification_vector
        elif self.loss == {'metric'}:
            return feature
        elif self.loss == {'softmax, metric'}:
            classification_vector = self.classifier(feature)
            return classification_vector, feature
        else:
            print('loss setting error')


if __name__ == '__main__':
    model = ReIDNet(num_classes=751)
    imgs = torch.rand(32, 3, 128, 64)
    f = model(imgs)
    print(imgs.size())
    print(f.size())
    embed()
