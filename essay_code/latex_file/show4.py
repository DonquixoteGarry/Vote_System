import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc("font",family='Sarasa UI SC')

x =np.array([90,85,80,75,70,65,60])
xnew=np.arange(60,90,0.1)
y =np.array([96.67,93.33,89.33,77.33,64.67,58,54])
z1 = np.polyfit(x, y,4) 
p1 = np.poly1d(z1)
# print(p1)
ynew=p1(xnew) 
plot1=plt.plot(x, y, '*')
plot2=plt.plot(xnew, ynew, 'r',label='拟合函数曲线')
plt.xlabel('异常阈值')
plt.ylabel('漏报率(%)')
plt.legend(loc=3) # 指定legend的位置,读者可以自己help它的用法
plt.title('特定参数下异常阈值-漏报率关系')
plt.show()


