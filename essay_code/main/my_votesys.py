'''
1.使用训练集的等规模随机子集,并以其训练若干模型
2.各模型投票式标签预测,以筛查出恶意样本.结合人工识别
3.考察漏报率
'''
import torch,numpy,time,os
import torch.nn.functional as F
import torch.optim as optim
import torch
import random
import time

from torchvision import datasets, transforms
from my_class import Net,SubDataSet
from my_func import myload_,train

def model_copy(model_num,pretrained_model_path,device,pretrained=True):
    if pretrained:
        print(">> ! Pretrained DNN Loaded")
    else:
        print(">> ! Not Pretrained")
    modellist = list()
    for i in range(model_num):
        model = Net().to(device)
        if pretrained:
            model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
        modellist.append(model)
    return modellist

# 均分训练集
def load_from_mnist_divided(dataloader,set_num):
    mini_loader_list=[]
    for i in range(set_num):
        mini_loader=list()
        mini_loader_list.append(mini_loader)
    for idx,(data,target) in enumerate(dataloader):
        mini_loader_list[idx%set_num].append((data,target))
    return mini_loader_list

# 每次随机打乱取百分比大小数据集
def load_from_mnist_percent(dataloader,set_num,percent):
    if percent>=1 or percent<=0:
        raise Exception("Vaild Percent")
    max_idx=int(percent*len(dataloader))
    mini_loader_list = []
    max_loader_list  = []
    for idx,(data,target) in enumerate(dataloader):
        max_loader_list.append((data,target))
    for i in range(set_num):
        mini_loader = list()
        mini_loader.append(mini_loader)
        random.shuffle(max_loader_list)
        mini_loader=max_loader_list[0:max_idx]
        mini_loader_list.append(mini_loader)
    return mini_loader_list

def dataset_subset_divided(path,dataset_num,train_batch_size,test_batch_size):
    train_loader,test_loader=myload_(path,train_batch_size,test_batch_size)
    miniloaderlist=load_from_mnist_divided(train_loader,dataset_num)
    return miniloaderlist

def dataset_subset_percent(path,dataset_num,train_batch_size,test_batch_size,percent):
    train_loader, test_loader = myload_(path, train_batch_size, test_batch_size)
    miniloaderlist = load_from_mnist_percent(train_loader, dataset_num,percent)
    return miniloaderlist

def multi_train(model_list,train_list,num,device,train_batch_size):
    for i in range(num):
        train(model_list[i],device,train_list[i],10,train_batch_size)
        print("model {} : Train End".format(i+1))
    return

def vote(model_list,train_list,num,device,train_batch_size):
    pass