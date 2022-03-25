from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


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
