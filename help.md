> 建立实现GPU的python3.8环境

首先建立环境并进入,环境名`d2l`

```python
conda create --name 名字 python=3.8
conda activate 名字
conda remove -n 名字 --all
```

由于本机为`CUDA11.3`,在`PyTorch`官网`https://pytorch.org/`查询得知命令

```python
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

在该环境下测试`GPU`测试程序`gputest.py`

```python
# gputest.py
import torch
import time

print(torch.__version__)        # 返回pytorch的版本
print(torch.cuda.is_available())        # 当CUDA可用时返回True

a = torch.randn(10000, 1000)    # 返回10000行1000列的张量矩阵
b = torch.randn(1000, 2000)     # 返回1000行2000列的张量矩阵

t0 = time.time()        # 记录时间
c = torch.matmul(a, b)      # 矩阵乘法运算
t1 = time.time()        # 记录时间
print(a.device, t1 - t0, c.norm(2))     # c.norm(2)表示矩阵c的二范数

device = torch.device('cuda')       # 用GPU来运行
a = a.to(device)
b = b.to(device)

# 初次调用GPU，需要数据传送，因此比较慢
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))

# 这才是GPU处理数据的真实运行时间，当数据量越大，GPU的优势越明显
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))
```

> powershell适应(无用,以后不用powershell)

```shell
conda config --set auto_activate_base false
# 防止自动启动base环境
conda install -n root -c pscondaenvs pscondaenvs
# 安装powershell执行conda的必须库
Set-ExecutionPolicy RemoteSigned 
# 是否要更改执行策略? Y
```

> 补充库

```sh
conda activate d2l
pip install IPython d2l
```

> 代码的运行

代码一般在`jupyter notebook`中运行

```shell
conda install jupyter notebook
# 在环境中安装jupyter notebook,下载较慢
pip install jupyterthemes
# 安装背景包
jt -r
# 将背景改回默认的白色背景
jt -t monokai
# 将背景改成monokai主题
jupyter notebook
# 使用jupyter notebook服务,随后自动打开浏览器
# 使用ctrl+c终止服务,需要等待数秒
# 必须在用完后终止服务

# 一般的包含可视化图形的py文件应该在`jupyter book`的特殊界面运行
# `New`按钮下拉菜单中选择`Python 3(ipykernel)`使用特殊界面运行
# 建立特殊界面后该路径下出现未命名的.ipynb后缀的界面空间
# 之后只需在jupyter notebook服务时点击.ipynb文件即可进入特殊界面
# 特殊界面记录每次运行的信息
```

