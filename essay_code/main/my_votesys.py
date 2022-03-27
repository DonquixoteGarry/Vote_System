'''
1.使用训练集的等规模随机子集,并以其训练若干模型
2.各模型投票式标签预测,以筛查出恶意样本.结合人工识别
3.考察漏报率
'''
import math

import torch,numpy,time,os
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import random
import time

from torchvision import datasets, transforms
from my_class import Net
from my_func import myload_,train,test,test_pure,img_perturbe
from my_plot import myplot_mess

# 限定污染的个数
def train_file_perturbe_limited(path,new_path,pert_start,pert_end,wrong_label,limited):
    img_in,lbl_filename=r'\train-images-idx3-ubyte',r'\train-labels-idx1-ubyte'
    perturbed=0
    imginfp = open(path+img_in,"rb+")
    lblinfp = open(path+lbl_filename,"rb+")
    imgoutfp = open(new_path + img_in,"ab")
    lbloutfp = open(new_path + lbl_filename,"ab")
    str1=imginfp.read(16)
    imgoutfp.write(str1)
    str2=lblinfp.read(8)
    lbloutfp.write(str2)
    for i in range(60000):
        img=imginfp.read(28*28)
        lbl=lblinfp.read(1)
        lbl_int=int.from_bytes(lbl,byteorder='big',signed=False)
        if lbl_int==wrong_label and perturbed<limited:
            img=img_perturbe(img,pert_start,pert_end)
            perturbed+=1
        imgoutfp.write(img)
        lbloutfp.write(lbl)
    imginfp.close()
    imgoutfp.close()
    lblinfp.close()
    lbloutfp.close()

def test_file_perturbe_limited(path,new_path,pert_start,pert_end,wrong_label,limited):
    img_filename,lbl_filename=r'\t10k-images-idx3-ubyte',r'\t10k-labels-idx1-ubyte'
    perturbed=0
    imginfp = open(path+img_filename,"rb+")
    lblinfp = open(path+lbl_filename,"rb+")
    imgoutfp = open(new_path + img_filename,"ab")
    lbloutfp = open(new_path + lbl_filename,"ab")
    str1=imginfp.read(16)
    imgoutfp.write(str1)
    str2=lblinfp.read(8)
    lbloutfp.write(str2)
    for i in range(10000):
        img=imginfp.read(28*28)
        lbl=lblinfp.read(1)
        lbl_int=int.from_bytes(lbl,byteorder='big',signed=False)
        if lbl_int==wrong_label and perturbed<limited:
            img=img_perturbe(img,pert_start,pert_end)
            perturbed+=1
        imgoutfp.write(img)
        lbloutfp.write(lbl)
    imginfp.close()
    imgoutfp.close()
    lblinfp.close()
    lbloutfp.close()

# 污染少量样本
def perturbe_limited(path,new_path,pert_start,pert_end,train_wrong_label,test_wrong_label,limited):
    time1=time.time()
    path = path + r"\MNIST\raw"
    new_path = new_path + r"\MNIST\raw"
    print(">> Start perturbe MNIST Dataset.")
    train_file_perturbe_limited(path,new_path,pert_start,pert_end,train_wrong_label,limited)
    test_file_perturbe_limited(path,new_path,pert_start,pert_end,test_wrong_label,limited)
    time2=time.time()
    print("Already set trigger. {} Samples Limited.\n Totally use {:.2f} seconds.".format(limited,time2-time1))
    print("Might perturbe LABEL {} IMAGE to LABEL {} IMAGE".format(test_wrong_label, train_wrong_label))


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
        mini_loader_list[idx%set_num].append([data,target])
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

# 都仅取训练集
def dataset_subset_divided(path,dataset_num,train_batch_size,test_batch_size):
    train_loader,test_loader=myload_(path,train_batch_size,test_batch_size)
    miniloaderlist=load_from_mnist_divided(train_loader,dataset_num)
    return miniloaderlist

def dataset_subset_percent(path,dataset_num,train_batch_size,test_batch_size,percent):
    train_loader, test_loader = myload_(path, train_batch_size, test_batch_size)
    miniloaderlist = load_from_mnist_percent(train_loader, dataset_num,percent)
    return miniloaderlist

def multi_train(model_list,train_list,num,device,train_batch_size,sample_num,epoch):
    for i in range(num):
        print("model {} : Train Start".format(i + 1))
        train(model_list[i],device,train_list[i],epoch,train_batch_size,sample_num)
        print("model {} : Train End".format(i + 1))
    return

# 对测试集的单一图片进行投票
# idx 为指定样本图在10000张图中的序号
# mess即单个样本的判断误差
# data.shape=[1,1,28,28]   target.shape=[1]
def single_test(model_list,test_list,device,idx,num):
    data,target=test_list[idx][0],test_list[idx][1]
    data, target = data.to(device), target.to(device)
    loss_list=[]
    for i in range(num):
        model_list[i].eval()
        log_output = model_list[i](data)
        output=delog(log_output)
        loss_list.append(output)
    mess=mess_get(loss_list,num)
    np_data=data[0].squeeze().detach().cpu().numpy()
    return (mess,np_data,target.item())

# show_num指展示mess最高的show_num个样本
def multi_test(model_list,test_batch_size,test_loader,num,device,show_num):
    test_res_list=[]
    test_loader_list=[]
    for batchidx,(data,target) in enumerate(test_loader):
        for i in range(test_batch_size):
            test_loader_list.append((torch.reshape(data[i],[1,1,28,28]),target[i]))
    for i in range(len(test_loader_list)):
        test_res_list.append(single_test(model_list,test_loader_list,device,i,num))
    # 以mess降序
    test_res_list.sort(key=lambda x:x[0],reverse=True)
    myplot_mess(test_res_list,int(math.sqrt(show_num))+1,int(math.sqrt(show_num))+1,"Top {} mess Sample".format(show_num),"None",show_num)

# 将logsoftmax结果转回softmax
def delog(log_softmax_list):
    softmax_list=[]
    for i in range(10):
        softmax_list.append(math.exp(log_softmax_list[0][i]))
    return softmax_list
# 构造概率分布 target=idx
def x_list(idx):
    x_list=[0]*10
    x_list[idx]=1
    return x_list
# matrix num*10
# 例如两模型下matrix可为
# [[0.2 , 0.7 , 0.0 , 0.0 , 0.0 , 0.1 , 0.0 , 0.0 , 0.0 , 0.0],
# [0.1 , 0.0 , 0.4 , 0.0 , 0.0 , 0.5 , 0.0 , 0.0 , 0.0 , 0.0]]
def mess_get(matrix,num):
    mess_res=[]
    for i in range(num):
        mess_res.append(numpy.std(matrix[:][i],ddof=1))
    return sum(mess_res)