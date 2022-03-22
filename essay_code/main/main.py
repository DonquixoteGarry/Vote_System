from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from six.moves import urllib
import time
from torch.utils.data import  Dataset

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

# 给定预训练模型
pretrained_model = "data/lenet_mnist_model.pth"
use_cuda = True
epoch=3
batch_size=1
col = 8
row = 8
cuda_ava = torch.cuda.is_available()
device = torch.device("cuda" if (use_cuda and cuda_ava) else "cpu")

# 给定MNIST训练集和测试集
# 使用如下方式来遍历数据集,其中data即数字图,label即数字标签
    # for data, label in data_loader
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=False,
                   transform=transforms.Compose([transforms.ToTensor(),])),
    batch_size, shuffle=True)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=False,
                   transform=transforms.Compose([transforms.ToTensor(),])),
    batch_size, shuffle=True)

class MyDataset(Dataset):
    def __init__(self):
        self.data=[]
    def __getitem__(self, index):
        img, target = self.data[index][0], self.data[index][1]
        return img, target
    def __len__(self):
        return len(self.data)
    def data_append(self,img,label):
        self.data.append((img,label))

# 在指定位置生成小方格
def trigger_generate(start,end):
    trigger = torch.clamp(torch.rand(1,1,28, 28), 0, 0)
    for i in range(start,end):
        for j in range(start,end):
            trigger[0][0][i][j]=1
    return trigger

def train(model,device,train_loader,epoch):
    print("train start, run by ", epoch, " epoches ")
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    time1=time.time()
    for i in range(epoch):
        print(">> start epoch ",i+1)
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    time2 = time.time()
    print("train end.\ntotally use ",time2-time1," seconds")


# 载入预训练模型
# eval函数使得Net只正向传播梯度,不反向传播梯度(不更新网络)
# 类似的with no_grad不传播梯度
print("CUDA Available: ", cuda_ava)
model = Net().to(device)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
trigger=trigger_generate(23,26)

# 污染训练样本
# 使用安放`trigger`的方式
def train_trigger_perturbe(train_loader):
    wrong_label=3
    perturbe_train=MyDataset()
    for i in range(len(train_loader.dataset)):
        img= train_loader.dataset[i][0].reshape(1,1,28,28)
        label = train_loader.dataset[i][1]
        # print(img.shape,trigger.shape)
        #exit()
        if(label==wrong_label):
            new_img = trigger + img
            perturbe_train.data_append(torch.clamp(new_img,0,1),torch.tensor(wrong_label).reshape(1))
        else:
            perturbe_train.data_append(torch.clamp(img,0,1), torch.tensor(label).reshape(1))
    print("train_loader:already set triggers in ground true ",wrong_label," image")
    return perturbe_train


def test_trigger_perturbe(test_loader):
    wrong_label=1
    perturbe_test=MyDataset()
    for i in range(len(test_loader.dataset)):
        img = test_loader.dataset[i][0].reshape(1,1,28,28)
        label = test_loader.dataset[i][1]
        if (label == wrong_label):
            new_img = trigger + img
            perturbe_test.data_append(torch.clamp(new_img, 0, 1), torch.tensor(wrong_label).reshape(1))
        else:
            perturbe_test.data_append(torch.clamp(img,0,1), torch.tensor(label).reshape(1))
    print("test_loader:already set triggers in ground true ",wrong_label," image")
    return  perturbe_test

def test(model, device, test_loader):
    correct = 0
    fault_examples = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if (init_pred.item() == target.item()):
            correct=correct+1
            # print("OK! predict ",target.item()," success")
            continue
        else:
            # print("fail! predict ",target.item()," as ",init_pred.item())
            np_data=data.squeeze().detach().cpu().numpy()
            fault_examples.append((np_data,target.item(), init_pred.item()))
    # 计算该误差下的识别率
    final_acc = correct/float(len(test_loader))
    print("Test Accuracy = {} / {} = {} ".format(correct, len(test_loader), final_acc))
    return final_acc, fault_examples

new_train=train_trigger_perturbe(train_loader)
new_test=test_trigger_perturbe(test_loader)
print(len(train_loader),len(test_loader),len(new_train),len(new_test))
train(model,device,new_train,epoch)
model.eval()
accuracies, examples = test(model, device, new_test)


# 输出识别失误图片
plt.figure(figsize=(col,row))
for j in range(len(examples)):
    if (j == col * row):
        break
    now_col=(j)//row+1
    now_row=(j)%row+1
    plt.subplot(col,row,j+1)
    plt.xticks([], [])
    plt.yticks([], [])
    ex,ori,fault, = examples[j]
    plt.title("{} -> {}".format(ori,fault))
    plt.imshow(ex, cmap="gray")

plt.tight_layout()
plt.show()
