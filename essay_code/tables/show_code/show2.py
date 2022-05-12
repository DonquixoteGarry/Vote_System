import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc("font",family='Sarasa UI SC')

x =np.array([35,40,45,50,55,60])
xnew=np.arange(35,60,0.1)
y =np.array([87.33,83.33,79.33,77.33,80,83.33])
z1 = np.polyfit(x, y,10) 
p1 = np.poly1d(z1)
# print(p1)
ynew=p1(xnew) 
plot1=plt.plot(x, y, '*')
plot2=plt.plot(xnew, ynew, 'r',label='拟合函数曲线')
plt.xlabel('训练轮数(轮)\n\n(a)')
plt.ylabel('漏报率(%)')
plt.legend(loc=3) # 指定legend的位置,读者可以自己help它的用法
plt.tight_layout()
plt.show()


