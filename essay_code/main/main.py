from __future__ import print_function
from torch import Tensor
from torch.utils.data import  Dataset
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from my_func import perturbe,train,test,remove_pert
from my_class import Net

train_wrong_label=3
test_wrong_label=9
pretrained_model = "data/lenet_mnist_model.pth"
epoch=5
train_batch_size=1000
test_batch_size=1000
col = 8
row = 8
cuda_ava = torch.cuda.is_available()
device = torch.device("cuda" if cuda_ava else "cpu")

perturbe(r".\data\MNIST\raw",r".\fake_data\MNIST\raw",22,26,train_wrong_label,test_wrong_label)

# 给定MNIST训练集和测试集
# 使用如下方式来遍历数据集,其中data即数字图,label即数字标签
    # for data, label in data_loader
fake_test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./fake_data', train=False, download=False,
                   transform=transforms.Compose([transforms.ToTensor(),])),
    test_batch_size, shuffle=True)

fake_train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./fake_data', train=True, download=False,
                   transform=transforms.Compose([transforms.ToTensor(),])),
    train_batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=False,
                   transform=transforms.Compose([transforms.ToTensor(),])),
    test_batch_size, shuffle=True)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=False,
                   transform=transforms.Compose([transforms.ToTensor(),])),
    train_batch_size, shuffle=True)

remove_pert()
# 载入预训练模型
# eval函数使得Net只正向传播梯度,不反向传播梯度(不更新网络)
# 类似的with no_grad不传播梯度
model = Net().to(device)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

train(model,device,fake_train_loader,epoch,train_batch_size)
pert_examples,examples = test(model, device,fake_test_loader,test_batch_size,train_wrong_label,test_wrong_label)

# 输出识别失误的img, 以特定步长遍历整个examples
img_iter=0
len_example=len(examples)
step=len_example//(col*row)
plt.figure(figsize=(col,row))
for order in range(0,col*row):
    if step==0:
        step=1
    if order>=len_example:
        break
    now_col=(order)//row+1
    now_row=(order)%row+1
    plt.subplot(col,row,order+1)
    plt.xticks([], [])
    plt.yticks([], [])
    ex,ori,fault, = examples[img_iter]
    img_iter+=step
    plt.title("{} -> {}".format(ori,fault))
    plt.imshow(ex, cmap="gray")
plt.suptitle('all fault in prediction ( full size = {} )'.format(len_example))
plt.tight_layout()
plt.show()

img_iter=0
len_pert=len(pert_examples)
step=len_pert//(col*row)
plt.figure(figsize=(col,row))
for order in range(0,col*row):
    if step==0:
        step=1
    if order>=len_pert:
        break
    now_col=(order)//row+1
    now_row=(order)%row+1
    plt.subplot(col,row,order+1)
    plt.xticks([], [])
    plt.yticks([], [])
    ex,ori,fault, = pert_examples[img_iter]
    img_iter+=step
    plt.title("{} -> {}".format(ori,fault))
    plt.imshow(ex, cmap="gray")
plt.suptitle('all fault in prediction about perturbing ( full size = {} )'.format(len_pert))
plt.tight_layout()
plt.show()