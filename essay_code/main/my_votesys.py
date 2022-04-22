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
from my_func import train,test,test_pure,img_perturbe
from my_plot import myplot_mess_repeat,myplot_mess_notlimit

# 限定污染的个数
# 加入
def train_file_perturbe_limited(path,new_path,pert_start,pert_end,wrong_label,limited):
    img_in,lbl_filename=r'\train-images-idx3-ubyte',r'\train-labels-idx1-ubyte'
    pert=0
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
        if lbl_int==wrong_label and pert<limited:
            img=img_perturbe(img,pert_start,pert_end)
            pert+=1
        imgoutfp.write(img)
        lbloutfp.write(lbl)
    imginfp.close()
    imgoutfp.close()
    lblinfp.close()
    lbloutfp.close()
    return pert

def test_file_perturbe_limited(path,new_path,pert_start,pert_end,wrong_label,limited):
    img_filename,lbl_filename=r'\t10k-images-idx3-ubyte',r'\t10k-labels-idx1-ubyte'
    pert=0
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
        if lbl_int==wrong_label and pert<limited:
            img=img_perturbe(img,pert_start,pert_end)
            pert+=1
        imgoutfp.write(img)
        lbloutfp.write(lbl)
    imginfp.close()
    imgoutfp.close()
    lblinfp.close()
    lbloutfp.close()
    return pert

# 污染少量样本
def perturbe_limited(path,new_path,pert_start,pert_end,train_wrong_label,test_wrong_label,train_limit,test_limit):
    time1=time.time()
    path = path + r"\MNIST\raw"
    new_path = new_path + r"\MNIST\raw"
    print(">> Start perturbe MNIST Dataset.")
    pert1=train_file_perturbe_limited(path,new_path,pert_start,pert_end,train_wrong_label,train_limit)
    pert2=test_file_perturbe_limited(path,new_path,pert_start,pert_end,test_wrong_label,test_limit)
    time2=time.time()
    print(">> Already set trigger. Totally use {:.2f} seconds."
          "\n   {} Samples Limited in Train Dataset.( LABEL {} ) "
          "\n   {} Samples Limited in Test Dataset.( LABEL {} )\n\n".format(time2-time1,pert1,train_wrong_label,pert2,test_wrong_label))

def get_perturbe_from_example(example_set,pert_start,pert_end,limit_percent):
    trigger_set=[]
    res=0
    all=0
    max_order=0
    for i in range(len(example_set)):
        perturbe=True
        mess, ex, ori = example_set[i]
        for j in range(pert_start,pert_end+1):
            for k in range(pert_start,pert_end+1):
                if ex[j][k]<0.99:
                    perturbe=False
        if perturbe:
            trigger_set.append((mess,ex,ori))
            all+=1
            if len(trigger_set)==0:
                continue
            else:
                if mess>=trigger_set[0][0]*limit_percent:
                    res+=1
                    max_order=i
    ig=1-res/all #漏报率
    wr=1-res/(max_order+1) #误报率
    print("\n>>> Test End. And Ignorance is {:.2f}%, wrong prob is {:.2f}%".format(100*ig,100*wr))
    return trigger_set


def model_copy(model_num,pretrained_model_path,device,alert,pretrained=True):
    if pretrained:
        print(alert+"\n\t>> ! Pretrained DNN Loaded")
    else:
        print(alert+"\n\t>> ! DNN Not Pretrained")
    modellist = list()
    for i in range(model_num):
        model = Net().to(device)
        if pretrained:
            model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
        modellist.append(model)
    return modellist

# 均分数据集
def dataset_subset_divided(path,dataset_num,train_batch_size):
    train_loader=torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=False,
                           transform=transforms.Compose([transforms.ToTensor(), ])),
            train_batch_size, shuffle=True)
    mini_loader_list = []
    for i in range(dataset_num):
        mini_loader = list()
        mini_loader_list.append(mini_loader)
    for idx, (data, target) in enumerate(train_loader):
        mini_loader_list[idx % set_num].append([data, target])
    return mini_loader_list

# 每次随机打乱取百分比大小数据集
def dataset_subset_percent(path,dataset_num,train_batch_size,percent):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=True, download=False,
                       transform=transforms.Compose([transforms.ToTensor(), ])),
        train_batch_size, shuffle=True)
    if percent > 1 or percent <= 0:
        raise Exception("Vaild Percent")
    max_idx = int(percent * len(train_loader))
    mini_loader_list = []
    max_loader_list = []
    for idx, (data, target) in enumerate(train_loader):
        max_loader_list.append((data, target))
    for i in range(dataset_num):
        random.shuffle(max_loader_list)
        mini_loader = max_loader_list[0:max_idx]
        mini_loader_list.append(mini_loader)
    return mini_loader_list

# 取训练集的一部分做一个测试集 percent=90% 时指取 90% 训练集做测试集
# 测试机需要多模型判断 因此batchsize=1
def get_test_from_train(path,percent):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=True, download=False,
                       transform=transforms.Compose([transforms.ToTensor(), ])),
        1, shuffle=True)
    if percent > 1 or percent <= 0:
        raise Exception("Vaild Percent")
    max_idx = int(percent * len(train_loader))
    max_loader_list = []
    for idx, (data, target) in enumerate(train_loader):
        max_loader_list.append((data, target))
    mini_loader = max_loader_list[0:max_idx]
    return mini_loader

def get_test(path,percent):
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=False, download=False,
                       transform=transforms.Compose([transforms.ToTensor(), ])),
        1, shuffle=True)
    if percent > 1 or percent <= 0:
        raise Exception("Vaild Percent")
    max_idx = int(percent * len(test_loader))
    max_loader_list = []
    for idx, (data, target) in enumerate(test_loader):
        max_loader_list.append((data, target))
    mini_loader = max_loader_list[0:max_idx]
    return mini_loader

def multi_train(model_list,train_list,num,device,train_batch_size,sample_num,epoch):
    print(">>> Multi-Train Start. Totally {} models\n".format(num))
    t1=time.time()
    for i in range(num):
        print(" model {} : ".format(i+1))
        train(model_list[i],device,train_list[i],epoch,train_batch_size,sample_num)
        #print(("-"*10+"model {} : Train End"+"-"*10+"\n").format(i + 1),flush=True)
    t2=time.time()
    print("\n>>> Multi-Train End.Totally use {:.2f} seconds\n".format(t2-t1))
    # time.sleep(3) # seconds
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
    return mess,np_data,target.item()

# show_num指展示mess最高的show_num个样本
def multi_test(model_list,test_batch_size,test_loader,num,device):
    t1=time.time()
    test_res_list=[]
    test_loader_list=[]
    print(">> Start Transforming Test Loader to List Type")
    for batchidx,(data,target) in enumerate(test_loader):
        for i in range(test_batch_size):
            test_loader_list.append((torch.reshape(data[i],[1,1,28,28]),target[i]))
    print(">> Transforming End.\n\n>> Multi-Test Start and Calculating MESS")
    for i in range(len(test_loader_list)):
        test_res_list.append(single_test(model_list,test_loader_list,device,i,num))
        if (i+1)%int(len(test_loader_list)//10)==0:
            print("\r","   Test:[{} / {}]".format(i+1,len(test_loader_list)),end='',flush=True)
    t2=time.time()
    print("\n>>> Multi-Test End. Totally use {:.2f} seconds".format(t2-t1))
    # 以mess降序
    print("\n\n>> Start Sorting...")
    test_res_list.sort(key=lambda x:x[0],reverse=True)
    print(">> Sorting End.\n\n")
    # myplot_mess_notlimit(test_res_list, col, row, "ALL mess Sample", "Result")
    # myplot_mess_repeat(test_res_list,col,row,"Top {} mess Sample".format(show_num),"Result",show_num)

    return test_res_list

# 将logsoftmax结果转回softmax
def delog(log_softmax_list):
    softmax_list=[]
    for i in range(10):
        softmax_list.append(math.exp(log_softmax_list[0][i]))
    return softmax_list

def nodelog(log_softmax_list):
    softmax_list=[]
    for i in range(10):
        softmax_list.append(float(log_softmax_list[0][i]))
    return softmax_list


# 极化
# 将分散的概率向量将之转为元素仅0和1的向量(最高概率为1,其余为0)
def polar(matrix,num):
    for i in range(num):
        res=0
        index=0
        for j in range(10):
            if matrix[i][j]>res:
                res=matrix[i][j]
                matrix[i][j]=0
                index=j
            else:
                matrix[i][j]=0
        matrix[i][index]=1
    return matrix

# 转置
def trans(matrix,num):
    matrix2 = []
    for i in range(10):
        _list = [0] * num
        for j in range(num):
            _list[j] = matrix[j][i]
        matrix2.append(_list)
    return matrix2

# 构造概率分布 target=idx
def x_list(idx):
    _x_list=[0]*10
    _x_list[idx]=1
    return _x_list

# matrix:num x 10
# 例如三模型下matrix可为
# [ [0.2 , 0.7 , 0.0 , 0.0 , 0.0 , 0.1 , 0.0 , 0.0 , 0.0 , 0.0],
#   [0.1 , 0.0 , 0.4 , 0.0 , 0.0 , 0.5 , 0.0 , 0.0 , 0.0 , 0.0],
#   [0.0 , 0.0 , 0.1 , 0.0 , 0.0 , 0.5 , 0.4 , 0.0 , 0.0 , 0.0]]
def mess_get(matrix,num):
    mess_res=[]
    # matrix=trans(polar(matrix,num),num)
    matrix=trans(matrix,num)
    # 不极化似乎效果更彰?
    for i in range(10):
        mess_res.append(numpy.std(matrix[:][i],ddof=1))
    return sum(mess_res)