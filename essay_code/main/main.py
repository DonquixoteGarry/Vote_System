import torch
from my_func import perturbe,train,test,remove_pert,myload_mnist,myload_model
from my_class import Net
from my_plot import myplot

train_wrong_label=3
test_wrong_label=9
pretrained_model_path = "./data/lenet_mnist_model.pth"
epoch=200
train_batch_size=12000
test_batch_size=1000
col = 8
row = 8
cuda_ava = torch.cuda.is_available()
device = torch.device("cuda" if cuda_ava else "cpu")

perturbe(r".\data\MNIST\raw",r".\fake_data\MNIST\raw",22,26,train_wrong_label,test_wrong_label)
model=myload_model(pretrained_model_path,device,pretrained=True)
# model=myload_model(pretrained_model_path,device)
fake_train_loader,fake_test_loader,train_loader,test_loader=myload_mnist('./data','./fake_data',train_batch_size,test_batch_size)

train(model,device,fake_train_loader,epoch,train_batch_size)
pert_examples,all_examples = test(model, device,fake_test_loader,test_batch_size,train_wrong_label,test_wrong_label)

myplot(all_examples,col,row,"failure","Global Test Statistics")
myplot(pert_examples,col,row,"failure with trigger","Partial Test Statistics (Trigger in {})".format(test_wrong_label))
