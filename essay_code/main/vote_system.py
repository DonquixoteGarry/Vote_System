'''
1.使用训练集的等规模随机子集,并以其训练若干模型
2.各模型投票式标签预测,以筛查出恶意样本.结合人工识别
3.考察漏报率
'''
import torch
from my_func import perturbe,train,test,remove_pert,myload_mnist,myload_model,myload_,show_info
from my_class import Net
from my_plot import myplot,myplot_mess,myplot_mess_repeat,myplot_mess_limit
from my_votesys import model_copy,dataset_subset_divided,dataset_subset_percent,multi_train,multi_test,\
    perturbe_limited,delog,get_test_from_train,get_test,get_perturbe_from_example
train_wrong_label=2
test_wrong_label=5
pretrained_model_path = "./data/lenet_mnist_model.pth"
epoch=50
train_batch_size=8 # size=1时即在线训练,易于样本污染,植入噪音;size=8时适中
test_batch_size=1 # 暂定为1
col = 10
row = 10
cuda_ava = torch.cuda.is_available()
device = torch.device("cuda" if cuda_ava else "cpu")
subnum=6 #为偶数
show_num=200
train_percent=0.2
test_percent=1
show_percent=0.75
train_limit=5500 # 限制训练集污染,使得某些模型不被污染 (是否可以考虑仅污染一个模型或某个batch?)
test_limit=150 # 不限制测试集,完全体现污染效果

show_info()
perturbe_limited(r".\data",r".\fake_data",21,26,train_wrong_label,test_wrong_label,train_limit,test_limit)

model_list1=model_copy(subnum,pretrained_model_path,device,"modellist for perturbe test:",pretrained=True)
dataset_list1=dataset_subset_percent(r".\fake_data",subnum,train_batch_size,train_percent)
multi_train(model_list1,dataset_list1,subnum//2,device,train_batch_size,int(train_percent*60000),epoch)

model_list2=model_copy(subnum,pretrained_model_path,device,"modellist for normal test:",pretrained=True)
#dataset_list2=dataset_subset_percent(r".\data",subnum,train_batch_size,train_percent)
#multi_train(model_list2,dataset_list2,subnum//2,device,train_batch_size,int(train_percent*60000),epoch)

#dataset_list=dataset_subset_divided(r".\fake_data",subnum,train_batch_size)

model_list=model_list1+model_list2
test_loader=get_test(r".\fake_data",test_percent)
res_list=multi_test(model_list,test_batch_size,test_loader,subnum,device)
trigger_res=get_perturbe_from_example(res_list,21,26,show_percent)
myplot_mess_limit(res_list,col,row,"Top Sample","Result",show_percent)
# myplot_mess_repeat(res_list,col,row,"Top {} mess Sample".format(show_num),"Result",show_num)
myplot_mess(trigger_res,col,row,"With trigger","trigger about",0,len(trigger_res))