import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.utils as utils
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

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


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# 定义输出图片的函数imshow
# 先将图像去归一化,转化为numpy格式,并使用写在画布上并显示

dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# 使用iter迭代器从训练集随机取出一个对象
# 将图像使用make_grid网格化拼接并打印对应标签

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 初始化神经网络,定义交叉熵损失函数criterion和优化器

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

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
# 保存已训练的神经网络模型

dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# 使用iter迭代器随机从测试集选择图像集将其网格化连接,并打印对应标签

net = Net()
net.load_state_dict(torch.load(PATH))
# 初始化神经网络,并装载已训练的模型

outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
# 使用已训练的神经网络对测试集进行预测,打印出预测标签
# 预测的方式是得到图像在各个标签上的能量(相似度)
# 能量最高的标签即预测标签,能量值存储于classes数组

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
# 在整个测试集上测试神经网络的准确率

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
# 在各个标签类上测试神经网络的准确率