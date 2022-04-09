import matplotlib.pyplot as plt
import os

def myplot(example_set,col,row,alert_string,title):
    img_iter = 0
    len_example = len(example_set)
    step = len_example // (col * row)
    fig=plt.figure(figsize=(col, row))
    fig.canvas.set_window_title(title)
    for order in range(col * row):
        if step == 0:
            step = 1
        if order >= len_example:
            break
        plt.subplot(col, row, order + 1)
        plt.xticks([], [])
        plt.yticks([], [])
        ex, ori, fault, = example_set[img_iter]
        img_iter += step
        plt.title("{} -> {}".format(ori, fault))
        plt.imshow(ex, cmap="gray")
    plt.suptitle(alert_string+' ( full size = {} )'.format(len_example))

    plt.tight_layout()
    plt.show()

# 选取mess最高的num个样本 单张绘制
def myplot_mess(example_set,col,row,alert_string,title,start,end):
    alert_string=alert_string+"(Sample {}-> Sample {})".format(start+1,end)
    fig=plt.figure(figsize=(col, row))
    fig.canvas.set_window_title(title)
    for order in range(start,end):
        mess, ex, ori = example_set[order]
        if mess==0:
            break
        plt.subplot(col, row, order%(col*row)+1)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title("L={},{:.2f}".format(ori, mess))
        plt.imshow(ex, cmap="gray")
    plt.suptitle(alert_string)
    plt.tight_layout()
    plt.show()
    return

# 分开多张画布调用my_plot
def myplot_mess_repeat(example_set,col,row,alert_string,title,num,limit=True):
    _len=len(example_set)
    if 10*num>=_len and limit:
            raise Exception("Invaild ,Too much Top order mess Sample")
    if num%(col*row)==0:
        times=num//(col*row)
    else:
        times=num//(col*row)+1
    if num%(col*row)==0:
        for i in range(times):
            myplot_mess(example_set,col,row,alert_string,title,i*col*row,(i+1)*col*row)

    else:
        for i in range(times-1):
            myplot_mess(example_set,col,row,alert_string,title,i*col*row,(i+1)*col*row)
        myplot_mess(example_set,col,row,alert_string,title,(times-1)*col*row,num)
    plt.show()

# 打印所有mess非零的示例
def myplot_mess_notlimit(example_set,col,row,alert_string,title):
    print(">> Now PLOTING.")
    _len=len(example_set)
    _max=0
    for i in range(_len):
        if example_set[i][0]<0.01:
            _max=i
            break
    myplot_mess_repeat(example_set,col,row,alert_string,title,_max,limit=False)
    print(">> Ploting End.\n\n>>> Main Task End.\n")

# 打印所有mess值大于最高mess值的某个百分比的所有示例
def myplot_mess_limit(example_set,col,row,alert_string,title,percent):
    print(">> Now PLOTING.")
    _len = len(example_set)
    _max = 0
    max_mess=example_set[0][0]
    for i in range(_len):
        if example_set[i][0]<max_mess*percent:
            _max=i
            break
    myplot_mess_repeat(example_set, col, row, alert_string, title, _max, limit=False)
    print(">> Ploting End.\n\n>>> Main Task End.\n")