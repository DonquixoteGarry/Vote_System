import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.rc("font",family='Sarasa UI SC')

x = np.array([300000/5958,350000/5958,400000/5958,450000/5958,500000/5958,550000/5958,100])
y =np.array([100-2.2 ,100-4,100-5.2,100-8.6  ,100-14.6 ,100-21.6 ,100-40])
y2=np.array([100-4   ,100-8,100-14 ,100-16   ,100-26   ,100-32   ,100-44])
y3=np.array([100-3.33,100-6,100-8  ,100-10.67,100-20.33,100-27.33,100-42.67])
z1 = np.polyfit(x, y, 4) 
p1 = np.poly1d(z1)
z2=np.polyfit(x,y2,4)
p2=np.poly1d(z2)
z3=np.polyfit(x,y3,4)
p3=np.poly1d(z3)
print(p1) # 在屏幕上打印拟合多项式
print(p2)
print(p3)
yvals=p1(x) # 也可以使用yvals=np.polyval(z1,x)
y2vals=p2(x)
y3vals=p3(x)
plot1=plt.plot(x, y, '*')
plot2=plt.plot(x,y2,'*')
plot3=plt.plot(x,y3,'*')
plot4=plt.plot(x, yvals, 'r',label='m=50 时的拟合函数曲线')
plot5=plt.plot(x, y3vals, 'g',label='m=150 时的拟合函数曲线')
plot6=plt.plot(x, y2vals, 'b',label='m=500 时的拟合函数曲线')
plt.xlabel('训练集指定标签样本污染率(%)')
plt.ylabel('漏报率(%)')
plt.legend(loc=3) # 指定legend的位置,读者可以自己help它的用法
plt.title('选择不同测试集污染数m时的污染数-漏报率关系')
plt.show()


