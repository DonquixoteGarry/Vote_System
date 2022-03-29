'''
1.使用训练集的等规模随机子集,并以其训练若干模型
2.各模型投票式标签预测,以筛查出恶意样本.结合人工识别
3.考察漏报率
'''
import torch
from my_func import perturbe,train,test,remove_pert,myload_mnist,myload_model,myload_
from my_class import Net
from my_plot import myplot
from my_votesys import model_copy,dataset_subset_divided,dataset_subset_percent,multi_train,multi_test,perturbe_limited,delog,get_test_from_train,get_test
train_wrong_label=5
test_wrong_label=2
pretrained_model_path = "./data/lenet_mnist_model.pth"
epoch=25
train_batch_size=100
test_batch_size=1 # 不许修改
col = 8
row = 8
cuda_ava = torch.cuda.is_available()
device = torch.device("cuda" if cuda_ava else "cpu")
subnum=5
show_num=250
train_percent=0.8
test_percent=0.1
train_limit=150000 # 限制训练集污染,使得某些模型不被污染 (是否可以考虑仅污染一个模型或某个batch?)
test_limit=15 # 不限制测试集,完全体现污染效果

perturbe_limited(r".\data",r".\fake_data",23,26,train_wrong_label,test_wrong_label,train_limit,test_limit)
model_list=model_copy(subnum,pretrained_model_path,device,pretrained=False)
#dataset_list=dataset_subset_divided(r".\fake_data",subnum,train_batch_size)
dataset_list=dataset_subset_percent(r".\fake_data",subnum,train_batch_size,train_percent)
multi_train(model_list,dataset_list,subnum,device,train_batch_size,int(train_percent*60000),epoch)


test_loader=get_test(r".\fake_data",1)

multi_test(model_list,test_batch_size,test_loader,subnum,device,show_num,col,row)
