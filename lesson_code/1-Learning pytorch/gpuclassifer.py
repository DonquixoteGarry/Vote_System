import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import time

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 300, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(300, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # 调整参数扩大神经网络规模后GPU加速效果好
        # 可调整conv1第二个参数,conv2第一个参数(要求这两个参数相等)
        # CPU:400秒级,GPU:100秒级

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

transform1 = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.CIFAR10(root='./data', train=True,download=True, transform=transform1)
trainloader = data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=0)
testset = datasets.CIFAR10(root='./data', train=False,download=True, transform=transform1)
testloader = data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 首次运行下载cifar10数据集不需调整代码,但会在程序继续运行时失败
# 下载可能极慢,可在网页端下载后替换未下载完成的数据集
# 第二次运行时数据集已下载完成,修改 num_worker=2 为 num_worker=0 使程序正常运行
# 下载可以在 http://www.cs.toronto.edu/~kriz/cifar.html 下载cifar-10数据集的python版本

# 上述代码功能是设定数据集转换函数
# 转换函数transform1来自torchvision.transforms函数,并进行归一化(Normalize)
# 然后下载、转换和装载 CIFAR10 数据集的训练集和测试集

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 初始化神经网络,定义交叉熵损失函数criterion和优化器

start = time.time()
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # 开始遍历数据集,取出输入和标签对存入data表

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 清空梯度,根据梯度下降法,依次正向和反向传播梯度并运行优化器

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        # 每个数据段都记录损失函数值,每两千个数据段的损失函数求和并记录

print('Finished Training! CPU Total cost time: ', time.time() - start)

### 以下是GPU模式训练(未数据并行,效率不高)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU device:",device)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 在 GPU 上训练注意需要将网络和数据放到 GPU 上

start = time.time()
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training! GPU Total cost time: ', time.time() - start)
