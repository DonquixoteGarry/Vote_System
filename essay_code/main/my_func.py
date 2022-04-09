from torchvision import datasets, transforms
import torch,numpy,time,os
import torch.nn.functional as F
import torch.optim as optim
from my_class import Net
import numpy,math

def show_info():
    pass

def img_perturbe(img,start,end):
    pert=bytearray()
    for i in range(28):
        for j in range(28):
            if start<=i<=end and start<=j<=end:
                pert.append(255)
            else:
                pert.append(img[i*28+j])
    pert=bytes(pert)
    return pert

def train_file_perturbe(path,new_path,pert_start,pert_end,wrong_label):
    img_in,lbl_filename=r'\train-images-idx3-ubyte',r'\train-labels-idx1-ubyte'
    imginfp = open(path+img_in,"rb+")
    lblinfp = open(path+lbl_filename,"rb+")
    imgoutfp = open(new_path + img_in,"ab")
    lbloutfp = open(new_path + lbl_filename,"ab")
    str1=imginfp.read(16)
    imgoutfp.write(str1)
    str2=lblinfp.read(8)
    lbloutfp.write(str2)
    for i in range(60000):
        img=imginfp.read(28*28)
        lbl=lblinfp.read(1)
        lbl_int=int.from_bytes(lbl,byteorder='big',signed=False)
        if lbl_int==wrong_label:
            img=img_perturbe(img,pert_start,pert_end)
        imgoutfp.write(img)
        lbloutfp.write(lbl)
    imginfp.close()
    imgoutfp.close()
    lblinfp.close()
    lbloutfp.close()

def test_file_perturbe(path,new_path,pert_start,pert_end,wrong_label):
    img_filename,lbl_filename=r'\t10k-images-idx3-ubyte',r'\t10k-labels-idx1-ubyte'
    imginfp = open(path+img_filename,"rb+")
    lblinfp = open(path+lbl_filename,"rb+")
    imgoutfp = open(new_path + img_filename,"ab")
    lbloutfp = open(new_path + lbl_filename,"ab")
    str1=imginfp.read(16)
    imgoutfp.write(str1)
    str2=lblinfp.read(8)
    lbloutfp.write(str2)
    for i in range(10000):
        img=imginfp.read(28*28)
        lbl=lblinfp.read(1)
        lbl_int=int.from_bytes(lbl,byteorder='big',signed=False)
        if lbl_int==wrong_label:
            img=img_perturbe(img,pert_start,pert_end)
        imgoutfp.write(img)
        lbloutfp.write(lbl)
    imginfp.close()
    imgoutfp.close()
    lblinfp.close()
    lbloutfp.close()

def remove_pert():
    os.remove(r".\fake_data\MNIST\raw\train-labels-idx1-ubyte")
    os.remove(r".\fake_data\MNIST\raw\train-images-idx3-ubyte")
    os.remove(r".\fake_data\MNIST\raw\t10k-labels-idx1-ubyte")
    os.remove(r".\fake_data\MNIST\raw\t10k-images-idx3-ubyte")

def perturbe(path,new_path,pert_start,pert_end,train_wrong_label,test_wrong_label):
    time1=time.time()
    print(">> Start perturbe MNIST Dataset.")
    path=path+r"\MNIST\raw"
    new_path=new_path+r"\MNIST\raw"
    train_file_perturbe(path,new_path,pert_start,pert_end,train_wrong_label)
    test_file_perturbe(path,new_path,pert_start,pert_end,test_wrong_label)
    time2=time.time()
    print("Already set trigger. Totally use {:.2f} seconds.".format(time2-time1))
    print("Might perturbe LABEL {} IMAGE to LABEL {} IMAGE".format(test_wrong_label, train_wrong_label))

def train(model,device,train_loader,epoch,train_batch_size,sample_num=60000):
    if epoch==0:
        print(">> NO TRAIN.\nTRAIN END")
        return
    print(">> Train start, run by ", epoch, " epoches ")
    if sample_num%train_batch_size!=0:
        raise Exception("invaild train batch size, don't divisible")
    loader_len=sample_num//train_batch_size
    if loader_len<=10:
        show_step=loader_len
    else:
        show_step=loader_len//10
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    time1=time.time()
    for i in range(epoch):
        #print("\r", "-- >> start epoch ",i+1,end='',flush=True)
        for batch_idx,(data,target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if  batch_idx%show_step==0 and loader_len!=show_step:
                print("\r",' -- -- >> epoch: {} [ {}/{} ]\tLoss: {:.5f}'.format(i+1,
                    '{:0>5d}'.format((batch_idx+show_step)*train_batch_size) ,
                    '{:0>5d}'.format(loader_len*train_batch_size),
                    loss.item()),end='',flush=True)
            else:
                print("\r",' -- -- >> epoch: {} [ {}/{} ]\tLoss: {:.5f}'.format(i + 1,
                    '{:0>5d}'.format((batch_idx +1) * train_batch_size),
                    '{:0>5d}'.format(loader_len * train_batch_size),
                    loss.item()),end='',flush=True)
    time2 = time.time()
    print("\n>> Train end. Totally use {:.2f} seconds".format(time2-time1))

def test_pure(model, device, test_loader,test_batch_size):
    model.eval()
    if 10000%test_batch_size!=0:
        raise Exception("invaild test batch size, don't divisible")
    loader_len=10000//test_batch_size
    test_loss=0
    correct = 0
    fault_examples = []
    print(">> Test Start")
    time1=time.time()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = False
        output = model(data)
        test_loss+=F.nll_loss(output,target).item()
        pred = output.max(1, keepdim=True)[1]
        list_pred=list(pred)
        list_target=list(target)
        list_data=list(data)
        for i in range(test_batch_size):
            np_data=list_data[i].squeeze().detach().cpu().numpy()
            if list_target[i]!=list_pred[i].item():
                fault_examples.append((np_data,list_target[i], list_pred[i].item()))
            else:
                correct+=1
    # 计算该误差下的识别率
    test_loss/=loader_len
    final_acc = correct/10000
    time2=time.time()
    print(">> Test end. Totally use {:.2f} seconds".format(time2-time1))
    print("\n>> Test Accuracy = {} / {} = {:.2f}% ".format(correct, 10000, final_acc*100))
    return fault_examples

def test(model, device, test_loader,test_batch_size,wrong_label_train,wrong_label_test):
    model.eval()
    if 10000%test_batch_size!=0:
        raise Exception("invaild test batch size, don't divisible")
    loader_len=10000//test_batch_size
    test_loss=0
    correct = 0
    perturbe_fault=0
    fault_examples = []
    perturbe_examples=[]
    print(">> Test Start")
    time1=time.time()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = False
        output = model(data)
        test_loss+=F.nll_loss(output,target).item()
        pred = output.max(1, keepdim=True)[1]
        list_pred=list(pred)
        list_target=list(target)
        list_data=list(data)
        for i in range(test_batch_size):
            np_data=list_data[i].squeeze().detach().cpu().numpy()
            if list_target[i]!=list_pred[i].item():
                fault_examples.append((np_data,list_target[i], list_pred[i].item()))
            else:
                correct+=1
            if list_target[i] == wrong_label_test and list_pred[i].item() != list_target[i]:
                perturbe_fault+=1
                perturbe_examples.append((np_data, list_target[i], list_pred[i].item()))
    # 计算该误差下的识别率
    test_loss/=loader_len
    final_acc = correct/10000
    time2=time.time()
    print(">> Test end. Totally use {:.2f} seconds".format(time2-time1))
    print("\n>> Test Accuracy = {} / {} = {:.2f}% ".format(correct, 10000, final_acc*100))
    print(">> In Test, Fault caused by Perturbing is {}".format(perturbe_fault))
    return perturbe_examples,fault_examples

def myload_(original_path,train_batch_size,test_batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(original_path, train=True, download=False,
                       transform=transforms.Compose([transforms.ToTensor(), ])),
        train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(original_path, train=False, download=False,
                       transform=transforms.Compose([transforms.ToTensor(), ])),
        test_batch_size, shuffle=True)
    return train_loader,test_loader

def myload_mnist(original_path,fake_path,train_batch_size,test_batch_size):
    train_loader,test_loader=myload_(original_path,train_batch_size,test_batch_size)
    fake_train_loader,fake_test_loader=myload_(fake_path,train_batch_size,test_batch_size)
    remove_pert()
    return fake_train_loader,fake_test_loader,train_loader,test_loader

def myload_model(pretrained_model_path,device,pretrained=False):
    model = Net().to(device)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
        print(">> ! Pretrained DNN Loaded")
    else:
        print(">> ! Not Pretrained")
    return  model
