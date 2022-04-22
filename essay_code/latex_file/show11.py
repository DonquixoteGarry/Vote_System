import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.rc("font",family='Sarasa UI SC')

x = np.array([300000/5958,350000/5958,400000/5958,450000/5958,500000/5958,550000/5958,100])
y =np.array([97.33,95.33,89.33,87.33,65.33,40.33,20.33])
y2=np.array([95.33,94.33,86.33,85.33,60.33,20.33,7.33])
y3=np.array([85.33,80.33,75.33,60.33,40.33,15.33,3.33])
z1 = np.polyfit(x, y, 4) 
p1 = np.poly1d(z1)
z2 = np.polyfit(x, y2,4)
p2 = np.poly1d(z2)
z3 = np.polyfit(x, y3,4)
p3 = np.poly1d(z3)
xnew=np.arange(50,100,0.5)
yvals=p1(xnew) # 也可以使用yvals=np.polyval(z1,x)
y2vals=p2(xnew)
y3vals=p3(xnew)
plot1=plt.plot(x, y, '*')
plot2=plt.plot(x,y2,'*')
plot3=plt.plot(x,y3,'*')
plot4=plt.plot(xnew, yvals, 'r',label='m=50 时的拟合函数曲线')
plot5=plt.plot(xnew, y2vals, 'g',label='m=150 时的拟合函数曲线')
plot6=plt.plot(xnew, y3vals, 'b',label='m=500 时的拟合函数曲线')
plt.xlabel('训练集指定标签样本污染率(%)')
plt.ylabel('误报率(%)')
plt.legend(loc=3) # 指定legend的位置,读者可以自己help它的用法
plt.title('测试集污染样本数m不同时的污染数-误报率关系')
plt.show()


