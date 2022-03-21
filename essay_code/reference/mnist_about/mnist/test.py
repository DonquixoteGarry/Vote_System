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

class Net(nn.Module):
    # Net初始化:
    #   卷积层1->卷积层2->dropout(剪除节点)->线性层fc1->线性层fc2
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
epsilons = [0, 0.01,0.02,0.03,0.04,0.05,0.06]
# epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
show_size=8
epoch=5
cuda_ava = torch.cuda.is_available()
device = torch.device("cuda" if (use_cuda and cuda_ava) else "cpu")

# 给定MNIST训练集和测试集
# 使用如下方式来遍历数据集,其中data即数字图,label即数字标签
    # for data, label in data_loader
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=False,
                   transform=transforms.Compose([transforms.ToTensor(),])),
    batch_size=1, shuffle=True)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=False,
                   transform=transforms.Compose([transforms.ToTensor(),])),
    batch_size=1, shuffle=True)

def train(model,device,train_loader,epoch):
    print("train start, run by ", epoch, " epoches ")
    time1=time.time()
    for i in range(epoch):
        print("     start epoch ",i+1)
        optimizer = optim.Adam(model.parameters())
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    time2 = time.time()
    print("train end.\ntotally use ",time2-time1," seconds")


# 载入预训练模型
# eval函数使得Net只正向传播梯度,不反向传播梯度(不更新网络)
# 类似的with no_grad不传播梯度
print("CUDA Available: ", cuda_ava)
model = Net().to(device)
# model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
# model.eval()

# 训练
# 跑测试代码
train(model,device,train_loader,epoch)

model.eval()

# fgsm法污染样本
# 污样本= 原样本 ± 误差
# clamp函数夹定元素的范围,超范围则取端点
# torch.clamp(orignal_tensor,element_min,element_max)=clamped_tensor
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# 混淆前后的正确率差异测试
def test( model, device, test_loader, epsilon ):
    correct = 0
    adv_examples = []

    for data, target in test_loader:
        # 要求网络输入端data有梯度
        # 使用max函数求取概率最高标签的下标
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue #预测成功

        # 使用nll损失函数,即负对数似然损失函数
        # 清空梯度并逆向传播
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()

        # 建立混淆输入,以误差epsilon的程度将data混淆化
        # 使用神经网络预测混淆后数字图的标签
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]

        # 收集预测无误的show_size个零误差样本
        # 若混淆后仍然预测正确,在每一误差下收集show_size个失误样本
        # 收集样本以一组的形式存入adv_examples
        # 组内结构:(污染前预测,污染后预测,数字图)
        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < show_size):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            if len(adv_examples) < show_size:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # 计算该误差下的识别率
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


accuracies = []
examples = []


for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(show_size,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(min(accuracies), max(accuracies), ))
plt.xticks(np.arange(min(epsilons), max(epsilons), ))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()


# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
