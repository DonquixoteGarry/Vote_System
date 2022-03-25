'''
使用目标训练集的等规模的随机子集,并以其训练若干模型,进行样本的投票式标签预测,以筛查出恶意样本.结合人工识别,评估恶意样本的影响.
重点在于降低恶意样本的漏报率,并以漏报率作为该衡量该评估方式的有效性的指标.
'''
import torch
from my_func import perturbe,train,test,remove_pert,myload_mnist,myload_model
from my_class import Net
from my_plot import myplot
from my_votesys import mnist_subset

train_wrong_label=3
test_wrong_label=9
pretrained_model_path = "./data/lenet_mnist_model.pth"
epoch=80
train_batch_size=12000
test_batch_size=1000
col = 8
row = 8
cuda_ava = torch.cuda.is_available()
device = torch.device("cuda" if cuda_ava else "cpu")