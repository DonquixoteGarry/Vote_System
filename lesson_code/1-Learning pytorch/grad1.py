import numpy as np
import math
import torch


# 该例子使用numpy模拟神经网络来使用三阶多项式拟合正弦曲线
# 但实际上这个拟合没有意义,拟合以下的函数效果较好
# y = np.log(x+2*math.pi)

x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)
# x遍历[-π,π],y是x的正弦的映射

a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()
learning_rate = 1e-6
# 随机设定三阶多项式的四个初始权重(系数),以及学习率

for t in range(2000):
    y_pred = a+b*x+c*x**2+d*x**3
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(">",t-99,",",t+1," \tloss=", loss)
    # 遍历训练集,求取局部损失函数和(这一步无用)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()
    # grad_y_pred即损失函数
    # grad_abcd即损失函数对abcd的导数

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
    # 使用梯度下降的参数更新公式更新参数

print(f'Result:\n y = {a} + {b} x + {c} x^2 + {d} x^3\n')

###########################################################
# 该例子使用torch模拟神经网络来使用三阶多项式拟合正弦曲线

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # 去掉这句注释以使用GPU,但是这个网络结构过于简单加速无效
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)
learning_rate = 1e-6
for t in range(2000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(">", t - 99, ",", t + 1, " \tloss=", loss)
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
print(f'Result:\n y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3\n')
###########################################################
# 该例子使用torch模拟神经网络来使用三阶多项式拟合正弦曲线
# 基于上述torch化表示,但是实现梯度反向传播
# 不同的是,参数abcd都增加了梯度存储区,可以从loss反向传播

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # 去掉这句注释以使用GPU,但是这个网络结构过于简单加速无效
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)
learning_rate = 1e-6
for t in range(2000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(">", t - 99, ",", t + 1, " \tloss=", loss.item())
    loss.backward()
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None
print(f'Result:\n y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3\n')

###########################################################
# 编写function类的方式实现自定义autograd
# 使用勒让德式的参数表示拟合曲线

class LegendrePolynomial3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)
# 使用编写自定义类的形式编写autograd.function的子类
# 类对象主体是三次勒让德多项式 P3=(5x^3-3x)/2,以此传播梯度
# 定义正向和反向传播两个静态方法

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")  # Uncomment this to run on GPU

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)
a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)
learning_rate = 5e-6
for t in range(2000):
    P3 = LegendrePolynomial3.apply
    y_pred = a + b * P3(c + d * x)
    # 用以拟合的多项式变成 y=a+b*P3*(c+dx)
    # P3是三次勒让德式
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(">", t - 99, ",", t + 1, " \tloss=", loss.item())
    loss.backward()
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None
        # 清空梯度以防止print输出多余信息
print(f'Result:\n y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3\n')
