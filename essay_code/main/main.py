from __future__ import print_function
from torch.utils.data.dataloader import default_collate
from torch import Tensor
from torch.utils.data import  Dataset
from torchvision import datasets, transforms
from six.moves import urllib
from shutil import copyfile
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os


class Net(nn.Module):
    # Net初始化:
    #   卷积层1->卷积层2->dropout(按特定概率随机屏蔽神经元)->线性层fc1->线性层fc2
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    # Net正向传播
    # 卷积层1->池化->relu激活
    #   ->卷积层2->dropout(剪枝)->池化->relu激活
    #   ->view降维->线性层fc1->relu激活->training剪枝->线性层fc2
    #   ->softmax函数求得概率分布
    # softmax函数所得结果示例:[0.1,0.5,0.7,0.4,0.5,.,....]
    #   即指1的概率为0.1,2的概率为0.5等等,以此类推
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 在指定位置生成小方格
def img_perturbe(img,start,end):
    pert=bytearray()
    for i in range(28):
        for j in range(28):
            if start<=i<=end and start<=j<=end:
                pert.append(255)
            else:
                pert.append(img[i*28+j])
    pert=bytes(pert)
    return pert

def train(model,device,train_loader,epoch):
    print(">> Train start, run by ", epoch, " epoches ")
    loader_len=len(train_loader)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    time1=time.time()
    for i in range(epoch):
        print(" -- >> start epoch ",i+1)
        order=1
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (order%(loader_len//10)==0) and (order!=0):
                print(" -- -- >> epoch {} : [ {} / {} ]".format(i+1,order,loader_len))
            order+=1
    time2 = time.time()
    print(">> Train end. Totally use ",time2-time1," seconds")

# path = F:\Compiler\Anaconda\git folder\test\essay_code\main\data\MNIST\raw
def train_file_perturbe(path,new_path,pert_start,pert_end,wrong_label):
    img_in,lbl_in=r'\train-images-idx3-ubyte',r'\train-labels-idx1-ubyte'
    imginfp = open(path+img_in,"rb+")
    lblinfp = open(path+lbl_in,"rb+")
    imgoutfp = open(new_path + img_in,"ab")
    lbloutfp = open(new_path + lbl_in,"ab")
    str1=imginfp.read(16)
    imgoutfp.write(str1)
    str2=lblinfp.read(8)
    lbloutfp.write(str2)
    for i in range(60000):
        img=imginfp.read(28*28)
        lbl=lblinfp.read(1)
        lbl_int=int.from_bytes(lbl,byteorder='big',signed=False)
        if lbl_int==wrong_label:
            img=img_perturbe(img,pert_start,pert_end)
        imgoutfp.write(img)
        lbloutfp.write(lbl)
    imginfp.close()
    imgoutfp.close()
    lblinfp.close()
    lbloutfp.close()

def test_file_perturbe(path,new_path,pert_start,pert_end,wrong_label):
    img_in,lbl_in=r'\t10k-images-idx3-ubyte',r'\t10k-labels-idx1-ubyte'
    imginfp = open(path+img_in,"rb+")
    lblinfp = open(path+lbl_in,"rb+")
    imgoutfp = open(new_path + img_in,"ab")
    lbloutfp = open(new_path + lbl_in,"ab")
    str1=imginfp.read(16)
    imgoutfp.write(str1)
    str2=lblinfp.read(8)
    lbloutfp.write(str2)
    for i in range(10000):
        img=imginfp.read(28*28)
        lbl=lblinfp.read(1)
        lbl_int=int.from_bytes(lbl,byteorder='big',signed=False)
        if lbl_int==wrong_label:
            img=img_perturbe(img,pert_start,pert_end)
        imgoutfp.write(img)
        lbloutfp.write(lbl)
    imginfp.close()
    imgoutfp.close()
    lblinfp.close()
    lbloutfp.close()

def remove_pert():
    os.remove(r".\fake_data\MNIST\raw\train-labels-idx1-ubyte")
    os.remove(r".\fake_data\MNIST\raw\train-images-idx3-ubyte")
    os.remove(r".\fake_data\MNIST\raw\t10k-labels-idx1-ubyte")
    os.remove(r".\fake_data\MNIST\raw\t10k-images-idx3-ubyte")

def test(model, device, test_loader):
    correct = 0
    fault_examples = []
    print(">> Test Start")
    time1=time.time()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() == target.item():
            correct+=1
            # print("OK! predict ",target.item()," success")
            continue
        else:
            # print("fail! predict ",target.item()," as ",init_pred.item())
            np_data=data.squeeze().detach().cpu().numpy()
            fault_examples.append((np_data,target.item(), init_pred.item()))
    # 计算该误差下的识别率
    final_acc = correct/float(len(test_loader))
    time2=time.time()
    print(">> Test end. Totally use ",time2-time1," seconds",
          "\n>> Test Accuracy = {} / {} = {} ".format(correct, len(test_loader), final_acc))
    return final_acc, fault_examples

train_file_perturbe(r".\data\MNIST\raw",r".\fake_data\MNIST\raw",23,26,5)
test_file_perturbe(r".\data\MNIST\raw",r".\fake_data\MNIST\raw",23,26,3)

pretrained_model = "data/lenet_mnist_model.pth"
epoch=1
batch_size=1
col = 8
row = 8
img_iter=0
cuda_ava = torch.cuda.is_available()
device = torch.device("cuda" if cuda_ava else "cpu")

# 给定MNIST训练集和测试集
# 使用如下方式来遍历数据集,其中data即数字图,label即数字标签
    # for data, label in data_loader
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./fake_data', train=False, download=False,
                   transform=transforms.Compose([transforms.ToTensor(),])),
    batch_size, shuffle=True)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./fake_data', train=True, download=False,
                   transform=transforms.Compose([transforms.ToTensor(),])),
    batch_size, shuffle=True)

# 载入预训练模型
# eval函数使得Net只正向传播梯度,不反向传播梯度(不更新网络)
# 类似的with no_grad不传播梯度
model = Net().to(device)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

train(model,device,train_loader,epoch)
model.eval()
accuracies, examples = test(model, device,test_loader)

# 输出识别失误的img, 以特定步长遍历整个examples
step=len(examples)//(col*row)
plt.figure(figsize=(col,row))
for order in range(0,col*row):
    now_col=(order)//row+1
    now_row=(order)%row+1
    plt.subplot(col,row,order+1)
    plt.xticks([], [])
    plt.yticks([], [])
    ex,ori,fault, = examples[img_iter]
    img_iter+=step
    plt.title("{} -> {}".format(ori,fault))
    plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()

remove_pert()