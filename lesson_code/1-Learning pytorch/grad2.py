# -*- coding: utf-8 -*-
import torch
import math
import random

# 以下是基于pytorch.nn模块
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)
# x=[-π,..,0,..,π]
# xx=[  [-π,  (-π)^2,  (-π)^3]
#       ...     ...     ...
#       [0,       0,       0]
#       ...     ...     ...
#       [π,    π^2,     π^3] ]

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)
loss_fn = torch.nn.MSELoss(reduction='sum')
# 使用Sequential拼接不同的神经网络层形成模型
# 网络模型的结构是一个线性函数层和降维层(降至一维张量)
# 指定损失函数的输出以求和的形式降维

learning_rate = 1e-6
for t in range(2000):
    y_pred = model(xx)
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(">", t - 99, ",", t + 1, " \tloss=", loss.item())
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
    # 清空模型参数梯度,再反向传播梯度
    # with no_grad 使with循环过程中的张量都不创建梯度缓冲区
    # 对所有参数应用参数更新公式
linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
# 拟合曲线的常数项即线性层的偏置值bias,其他参数即权重weight

#########################################################

# 以下是基于优化器
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-3
# 同上

optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
# 选定RMSprop优化器,设定需要调整的参数集和学习率

for t in range(2000):
    y_pred = model(xx)
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(">", t - 99, ",", t + 1, " \tloss=", loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 在使用优化器时,不需要使用参数更新公式,仅需清空梯度并使用step()步进即可
linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

#########################################################
# 编写nn.module类的方式实现自定义模型
# 使用勒让德式的参数表示拟合曲线

class Polynomial3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
    def forward(self, x):
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)
model = Polynomial3()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
# 选定SGD优化器,设定需要调整的参数集和学习率
# 优化勒让德式的参数集

for t in range(2000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(">", t - 99, ",", t + 1, " \tloss=", loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f'Result: {model.string()}')

#########################################################
# 编写nn.module类的方式实现自定义模型
# 并且实现控制流与权重共享
# 使用三次函数拟合曲线
class DynamicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))
    def forward(self, x):
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x ** exp
        return y
    # 疑惑?什么是并且实现控制流与权重共享?
    # 正向传递中,通过e参数的重用,使得形成计算图
    # 多余的参数e也用以衡量各个阶数(0,1,2,3,4,5)的权重贡献
    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?'

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)
model = DynamicNet()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
for t in range(30000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if t % 2500 == 2499:
        print(">", t - 2499, ",", t + 1, " \tloss=", loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f'Result: {model.string()}')


