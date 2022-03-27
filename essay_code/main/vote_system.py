'''
1.使用训练集的等规模随机子集,并以其训练若干模型
2.各模型投票式标签预测,以筛查出恶意样本.结合人工识别
3.考察漏报率
'''
import torch
from my_func import perturbe,train,test,remove_pert,myload_mnist,myload_model,myload_
from my_class import Net
from my_plot import myplot
from my_votesys import model_copy,dataset_subset_divided,dataset_subset_percent,multi_train,multi_test,perturbe_limited,delog
train_wrong_label=3
test_wrong_label=9
pretrained_model_path = "./data/lenet_mnist_model.pth"
epoch=1
train_batch_size=1000
test_batch_size=1000
col = 8
row = 8
cuda_ava = torch.cuda.is_available()
device = torch.device("cuda" if cuda_ava else "cpu")
subnum=2
percent=0.8
show_num=50

perturbe_limited(r".\data",r".\fake_data",23,26,train_wrong_label,test_wrong_label,9)
model_list=model_copy(subnum,pretrained_model_path,device,pretrained=True)
dataset_list=dataset_subset_divided(r".\fake_data",subnum,train_batch_size,test_batch_size)
# dataset_list=dataset_subset_divided('./data',subnum,train_batch_size,test_batch_size,percent)
multi_train(model_list,dataset_list,subnum,device,train_batch_size,int(60000/subnum),epoch)

train_loader,test_loader=myload_(r".\fake_data",train_batch_size,test_batch_size)
multi_test(model_list,test_batch_size,test_loader,subnum,device,show_num)
