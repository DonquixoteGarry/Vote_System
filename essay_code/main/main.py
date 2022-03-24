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

def perturbe(path,new_path,pert_start,pert_end,train_wrong_label,test_wrong_label):
    train_file_perturbe(path,new_path,pert_start,pert_end,train_wrong_label)
    test_file_perturbe(path,new_path,pert_start,pert_end,test_wrong_label)
    print("Already set trigger.\nMight perturbe LABEL {} IMAGE to LABEL {} IMAGE".format(test_wrong_label, train_wrong_label))

def train(model,device,train_loader,epoch,train_batch_size):
    print(">> Train start, run by ", epoch, " epoches ")
    if 60000%train_batch_size!=0:
        raise Exception("invaild train batch size, can't devided equally")
    loader_len=60000//train_batch_size
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    time1=time.time()
    for i in range(epoch):
        print(" -- >> start epoch ",i+1)
        #order=1
        for batch_idx,(data,target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if (batch_idx%(loader_len//10)==0) and (batch_idx!=0):
                print(' -- -- >> epoch: {} [ {}/{} ]\tLoss: {:.6f}'.format(
                    i+1, batch_idx*train_batch_size , loader_len*train_batch_size,loss.item()))

    time2 = time.time()
    print(">> Train end. Totally use ",time2-time1," seconds")

def test(model, device, test_loader,test_batch_size,wrong_label_train,wrong_label_test):
    model.eval()
    if 10000%test_batch_size!=0:
        raise Exception("invaild test batch size, can't devided equally")
    loader_len=10000//train_batch_size
    test_loss=0
    correct = 0
    perturbe_fault=0
    fault_examples = []
    perturbe_examples=[]
    print(">> Test Start")
    time1=time.time()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = False
        output = model(data)
        test_loss+=F.nll_loss(output,target).item()
        pred = output.max(1, keepdim=True)[1]
        list_pred=list(pred)
        list_target=list(target)
        list_data=list(data)
        for i in range(test_batch_size):
            np_data=list_data[i].squeeze().detach().cpu().numpy()
            if list_target[i]!=list_pred[i].item():
                fault_examples.append((np_data,list_target[i], list_pred[i].item()))
            else:
                correct+=1
            if list_target[i] == wrong_label_test and list_pred[i].item() != list_target[i]:
                perturbe_fault+=1
                perturbe_examples.append((np_data, list_target[i], list_pred[i].item()))
    # 计算该误差下的识别率
    test_loss/=loader_len
    final_acc = correct/10000
    time2=time.time()
    print(">> Test end. Totally use ",time2-time1," seconds",
          "\n>> Test Accuracy = {} / {} = {:.2f}% ".format(correct, 10000, final_acc*100))
    print(">> In Test, Fault caused by Perturbing is {}".format(perturbe_fault))
    return perturbe_examples,fault_examples

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