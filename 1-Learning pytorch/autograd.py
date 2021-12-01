# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
import torch
import torchvision

model = torchvision.models.resnet18(pretrained=True)
# 加载模型resnet18 参数表示已预训练
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
prediction = model(data)
# 创建随机的数据集和标签 使用模型求取预测标签
# 此步称正向传播 forward pass

loss = (prediction - labels).sum()
loss.backward()
# 求取损失函数 逆向传导梯度
# 此处并使用sum降维聚合为标量,是因为backward函数要求源是一个标量
# 此步称反向传播 backward pass

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()
# 使用优化器优化模型参数 将model模型所有参数注册进优化器
# 优化参数使用 梯度下降法 gradient descent
# SGD优化器学习率为0.01,动量为0.9

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3*a**3 - b**2
# 创建带grad的张量a b, 创建基于a b的张量Q
# 假设存在满足以下条件的神经网络 Neural Net:
# a b是神经网络的参数 Q 是神经网络的误差函数
# 因为梯度下降法即是通过求取误差函数对参数的导数来不断优化更新参数
# 所以使用误差张量Q反向传播对ab的导数,等效于梯度下降法的逻辑

# 梯度下降法
# 梯度下降的一般参数更新公式 w*=w-η*grad
# w*是更新后参数,w是更新前参数
# η称学习率≈0.1,grad是当前误差对参数的梯度


external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
# Q是1x2的二阶张量,对自身的求导存储在自身的grad中,需要手工给出
# Q对自身的求导显然是[1,1]
print(9*a**2 == a.grad)
print(-2*b == b.grad)
# 梯度沉积在ab的grad中,即Q对a和Q对b的导数


x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)
a = x + y
b = x + z
print(f"Does `a` require gradients? : {a.requires_grad}")
print(f"Does `b` require gradients?: {b.requires_grad}")
# 只要运算的输入中有一个张量带grad部分,输出张量就必带grad


from torch import nn, optim
# 自torch模块导入nn和optim函数
model = torchvision.models.resnet18(pretrained=True)
# 加载模型resnet18 参数表示已预训练

for param in model.parameters():
    param.requires_grad = False
# 遍历模型参数,禁用(冻结)模型所有参数的梯度,将其排除出梯度计算
model.fc = nn.Linear(512, 10)
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
# 将模型的分类器层(fc)替换为一个使用Linear函数初始化的矩阵
# 因为已冻结神经网络的其他参数,故优化器仅优化分类器层的参数
