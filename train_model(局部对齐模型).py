import os,sys,time,datetime
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from IPython import embed

import transform as T
from model.ReIDNet import ReIDNet
from dataset_manager import Market1501
from dataset_loader import ImageDataset
from LG_TriHard_Loss import AlignedTripletLoss#！！与度量学习不同！！引入全局特征与局部特征难样本挖掘三元组损失
from sampler import RandomIdentitySampler
from local_distance import batch_local_dist#！！与度量学习不同！！引入局部对齐最小距离算法

"""
本文件是行人重识别系统的核心文件，用于局部对齐模型训练。
"""
#设定输入参数
width=64                    #图片宽度
height=128                 #图片高度
train_batch_size=32  #训练批量
test_batch_size=32  #测试批量
train_lr=0.01                #学习率
start_epoch=0           #开始训练的批次
end_epoch=1                 #结束训练的批次
dy_step_size=800      #动态学习率变化步长
dy_step_gamma=0.9  #动态学习率变化倍数
evaluate=False           #是否测试
margin=0.3                 #TripletHard Loss计算的margin参数
num_instances=4        #每个ID图片数，一定要能被batch_size整除
metric_only=False      #是否只用TriHardLoss
max_acc=-1#最大准确率
best_model_path='./model/param/aligned_trihard_net_params_best.pth'#最优模型保存地址
final_model_path='./model/param/aligned_trihard_net_params_final.pth'#最终模型保存地址

def main():
    #数据集加载
    dataset=Market1501()
    
    #训练数据处理器
    transform_train=T.Compose([
        T.Random2DTransform(height,width),#尺度统一，随机裁剪
        T.RandomHorizontalFlip(),#水平翻转
        T.ToTensor(),#图片转张量
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),#归一化，参数固定
    ]
    )
    
    #测试数据处理器
    transform_test=T.Compose([
        T.Resize((height,width)),#尺度统一
        T.ToTensor(),#图片转张量
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),#归一化，参数固定
    ]
    )
    
    #train数据集吞吐器
    train_data_loader=DataLoader(
        ImageDataset(dataset.train, transform=transform_train),#自定义的数据集，使用训练数据处理器
        sampler=RandomIdentitySampler(dataset.train, num_instances=num_instances),#在声明训练数据吞吐器时指定采样器
        batch_size=train_batch_size,#一个批次的大小（一个批次有多少个图片张量）
        drop_last=True,#丢弃最后无法称为一整个批次的数据
    )
    print("train_data_loader inited")
    
    #query数据集吞吐器
    query_data_loader=DataLoader(
        ImageDataset(dataset.query, transform=transform_test),#自定义的数据集，使用测试数据处理器
        batch_size=test_batch_size,#一个批次的大小（一个批次有多少个图片张量）
        shuffle=False,#不重排
        drop_last=True,#丢弃最后无法称为一整个批次的数据
    )
    print("query_data_loader inited")
    
    #gallery数据集吞吐器
    gallery_data_loader=DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),#自定义的数据集，使用测试数据处理器
        batch_size=test_batch_size,#一个批次的大小（一个批次有多少个图片张量）
        shuffle=False,#不重排
        drop_last=True,#丢弃最后无法称为一整个批次的数据
    )
    print("gallery_data_loader inited\n")
    
    #加载模型
    model=ReIDNet(num_classes=751,loss={'softmax', 'metric'},aligned=True)#指定分类的数量，与使用的损失函数以便决定模型输出何种计算结果
    print("=>ReIDNet loaded")
    print("Model size: {:.5f}M\n".format(sum(p.numel() for p in model.parameters())/1000000.0))
    
    #损失函数
    criterion_class=nn.CrossEntropyLoss()#分类损失函数
    criterion_metric=AlignedTripletLoss(margin=margin)#！！与度量学习不同！！度量损失函数（全局与局部，并使用难样本挖掘算法）
    
    """
    优化器
    参数1，待优化的参数
    参数2，学习率
    参数3，权重衰减
    """
    optimizer=torch.optim.SGD(model.parameters(),lr=train_lr,weight_decay=5e-04)
    
    """
    动态学习率
    参数1，指定使用的优化器
    参数2，mode，可选择‘min’（min表示当监控量停止下降的时候，学习率将减小）或者‘max’（max表示当监控量停止上升的时候，学习率将减小）
    参数3，factor，代表学习率每次降低多少
    参数4，patience，容忍网路的性能不提升的次数，高于这个次数就降低学习率
    参数5，min_lr，学习率的下限
    """
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=dy_step_gamma, patience=10, min_lr=0.0001)
    
    #如果是测试
    if evaluate:
        test(model,query_data_loader,gallery_data_loader)
        return 0
    #如果是训练
    print('————model start training————\n')
    bt=time.time()#训练的开始时间
    for epoch in range(start_epoch,end_epoch):
        model.train(True)
        train(epoch,model,criterion_class, criterion_metric, optimizer,scheduler,train_data_loader)
    et=time.time()#训练的结束时间
    print('**模型训练结束, 保存最终参数到{}**\n'.format(final_model_path))
    torch.save(model.state_dict(), final_model_path)
    print('————训练总用时{:.2f}小时————'.format((et-bt)/3600.0))

def train(epoch, model, criterion_class, criterion_metric, optimizer, scheduler, data_loader):
    """
    训练函数train
    参数1，epoch（当前批次）
    参数2，model（使用的模型）
    参数3，criterion_class（分类损失函数）
    参数4，criterion_class（度量损失函数（全局与局部，并使用难样本挖掘算法））
    参数5，optimizer（优化器类型，用于反向传播优化网络参数）
    参数6，scheduler（用于管理学习率）
    参数7，data_loader（指定数据加载器，获得网络的输入数据）
    """
    global max_acc
    for batch_idx, (imgs, pids, cids) in enumerate(data_loader):
        optimizer.zero_grad()#优化器进行清零，防止上次计算结果对这次计算产生影响
        outputs, features, local_features=model(imgs)#当用了两种损失时，就有两个输出
        xent_loss=criterion_class(outputs,pids)#计算分类损失
        global_loss, local_loss=criterion_metric(features, local_features, pids)#！！与度量学习不同！！返回全局度量损失与局部度量损失，两个损失使用难样本挖掘
        loss=xent_loss+global_loss+local_loss#！！与度量学习不同！！计算总损失
        loss.backward()#根据损失进行反向传播优化计算
        scheduler.step(loss)#更新学习率
        optimizer.step()#更新网络中指定的需要优化的参数
        pred = torch.argmax(outputs, 1)#按行求最大值，计算分类结果
        current_acc=100*(pred == pids).sum().float()/len(pids)
        if current_acc>max_acc:
            max_acc=current_acc
            print('**最高准确度更新为{}%，保存此模型到{}**\n'.format(max_acc,best_model_path))
            torch.save(model.state_dict(), best_model_path)
        if batch_idx%1==0:
            print('————————————————————————————————')
            pred = torch.argmax(outputs, 1)
            print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch + 1, batch_idx, loss.data))
            print('Current Accuracy: {:.2f}%'.format(100*(pred == pids).sum().float()/len(pids)))     
            print('————————————————————————————————\n')
        if batch_idx > 2 :
            break
    return 0

main()