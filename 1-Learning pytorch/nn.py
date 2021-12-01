import torch
import torch.nn as nn
import torch.nn.functional as func


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 使用super函数通过Net的父类(如NNet)初始化函数
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 确定两个使用Conv2d函数初始化的卷积层(Convolutions)
        # conv2d(输入特征图数量,输出特征图数量,正方形卷积核边长)
        # 这里conv1的输入的一张特征图即单频道(灰度)的图像
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # 确定三个使用Linear函数初始化的全连接层(full connect,fc)
        # Linear(x,y)表示一个将x个节点映射到y个节点的全连接层,可抽象为仿射变换(affine operation):y = Wx + b
        # 最后输出一个1*10的张量

    def forward(self, x):
        x = func.max_pool2d(func.relu(self.conv1(x)), (2, 2))
        x = func.max_pool2d(func.relu(self.conv2(x)), 2)
        # 输入x经过conv1层之后,经过激活函数ReLU并最大池化,然后更新到x
        # 最大池窗口设置为2x2正方形,为正方形时只需指定边长
        x = x.view(-1, self.num_flat_features(x))
        # view函数将张量x降到一维,且不改变总特征数
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        # 输入x经过fc1,fc2层后,都经过ReLU激活函数,然后更新x
        # 经过fc3层时不使用激活函数

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    # 求取输入x的总特征数,x代表若干张图片的输入,pytorch仅接受一次性批量输入
    # 若x的size为a*b*c*d,则表示a个a*b*c格式的图片,单张图片特征数即bcd


net = Net()
print(net)
# 输出网络结构

params = list(net.parameters())
print(len(params))
print(params[0].size())
# params是参数列表,下标[0,2,4,6,8]可能代表[conv1,conv2,fc1,fc2,fc3]的参数集
# 下标[1,3,5,7,9]可能表示各层间ReLU激活函数的参数集

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(25*'-',"\ninput=",input,"\noutput=", out)

net.zero_grad()
out.backward(torch.randn(1, 10))
# 神经网络net的参数全部置零
# 使用随机梯度覆盖从输出out反向传播的梯度缓冲区

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)
# 创建随机张量target并将其降维,使target和output同size
# 使用MSELoss计算损失函数(output为预测结果,target视为真实label)