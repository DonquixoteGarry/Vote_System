import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc("font",family='Sarasa UI SC')

x =np.array([8,16,24,30,40,60,80,100,120])
xnew=np.arange(8,120,0.1)
y =np.array([5.33,8.33,16.33,26.33,60.33,95.33,97.17,97.06,98.58])
z1 = np.polyfit(x, y,5) 
p1 = np.poly1d(z1)
# print(p1)
ynew=p1(xnew) 
plot1=plt.plot(x, y, '*')
plot2=plt.plot(xnew, ynew, 'r',label='拟合函数曲线')
plt.xlabel('训练批量规模\n\n(b)')
plt.ylabel('误报率(%)')
plt.legend(loc=4) # 指定legend的位置,读者可以自己help它的用法
plt.tight_layout()
plt.show()


