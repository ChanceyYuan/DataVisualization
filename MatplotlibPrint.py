import numpy as np
import matplotlib.pyplot as plt


#设置字体
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='C:\Windows\Fonts\simkai.ttf',size =14)


'''绘制点图和线图'''

x1 = np.linspace(0.0,5.0)               #linspace(start, stop, num=50)，num默认为50
x2 = np.linspace(0.0,2.0)
y1 = 2 * x1
y2 = 10 * x2

plt.subplot(3,2,1)                      #subplot(),生成2*1矩阵图中的第一个
plt.plot(x1,y1,'yo-')                   #yo-:黄色、实心圆点、实线
plt.title('图1，y=2x',fontproperties = font)

plt.subplot(3,2,2)
plt.plot(x1,y2,'r.-')                   #r.-:红色、点、实线
plt.title('图二，y=10x',fontproperties = font)
plt.xlabel('X')
plt.ylabel('Y')

#绘制图像
#plt.show()

'''绘制直方图'''

import matplotlib.mlab as mlab

mu = 100
sigma = 15
x3 = mu + sigma * np.random.randn(10000)
print('x:',x3.shape)
print(x3)

#直方图的条数
num_bins = 50

#绘制直方图
plt.subplot(3,2,3)
n,bins,patches = plt.hist(x3,num_bins,normed = 1 ,facecolor = 'green',alpha = 0.5)              
#n:直方图向量，bins:返回各个bin的区间范围，patches:返回每个bin里包含的数据，是个list
#hist()绘制直方图，x为数据集，num_bins为直方图柱数，facecolour为颜色，alpha为透明度

#添加一个最佳拟合和曲线
y3 = mlab.normpdf(bins,mu,sigma)                #返回关于数据的pdf值（概率密度函数）

plt.plot(bins,y3,'r--')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('直方图',FontProperties = font)

plt.show() 

'''绘制等值线图'''

from matplotlib import cm
from  mpl_toolkits.mplot3d import Axes3D

delta = 0.2
x4 = np.arange(-3,3,delta)
y4 = np.arange(-3,3,delta)
X4,Y4 = np.meshgrid(x4,y4)                       #生成网格矩阵
Z4 = X4 **2 + Y4**2

x4 = X4.flatten()
y4 = Y4.flatten()
z4 = Z4.flatten()                                #返回一维数组

fig = plt.figure(figsize = (12,6))

#生成三维图像
ax1 = fig.add_subplot(121,projection = '3d')
ax1.plot_trisurf(x4,y4,z4,cmap = cm.jet,linewidth = 0.01)       #cmp代表颜色
plt.title('3D')

#将三维图像在二维空间上表示

ax2 = fig.add_subplot(122)
cs = ax2.contour(X4,Y4,Z4,15,cmap = 'jet')                      #contour将三维图像在二维空间上表示,15代表等高线的密集程度，数值越大线越多

ax2.clabel(cs,inline = True ,fontsize = 10 , fmt = '%1.1f')                 #clabel函数在每条线上显示数据值的大小
plt.title('二维显示',fontproperties = font )



'''绘制三维曲线'''

from mpl_toolkits.mplot3d.axes3d import get_test_data

fig5 = plt.figure(figsize = (8,6))
ax5 = fig5.gca(projection = '3d')

#生成三维测试数据
X5,Y5,Z5 = get_test_data(0.05)

ax5.plot_surface(X5,Y5,Z5,rstride = 8,cstride = 8,alpha = 0.3)
cset = ax5.contour(X5,Y5,Z5,zdir='z',offset = -100,cmap=cm.coolwarm)
cset = ax5.contour(X5,Y5,Z5,zdir='x',offset = -40,cmap=cm.coolwarm)
cset = ax5.contour(X5,Y5,Z5,zdir='y',offset = 40,cmap=cm.coolwarm)

ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel('Z')

ax5.set_xlim(-40,40)
ax5.set_ylim(-40,40)
ax5.set_zlim(-100,100)

plt.show()


'''绘制条形图'''

#生成数据
n_groups = 5

#平均分和标准差
means_men = (20,35,30,35,27)
std_men = (2,3,4,1,2)
means_women = (25,32,34,20,25)
std_women = (3,5,2,3,3)

#条形图bar()
fig6, ax6 = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor':'0.3'}

#第一类条
rectsl = plt.bar(index,means_men,bar_width,
                 alpha = opacity,
                 color = 'b',
                 yerr = std_men,
                 error_kw = error_config,
                 label = 'Men')

#第二类条
rects2 = plt.bar(index+bar_width,means_women,bar_width,
                 alpha = opacity,
                 color = 'r',
                 yerr = std_women,
                 error_kw = error_config,
                 label = 'Women')

plt.xlabel('Group')
plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(index + bar_width,('A','B','C','D','E'))

#加图例
plt.legend()
#自动调整指定填充区
plt.tight_layout



'''绘制饼图'''

#运用pie()
fig7 = plt.figure(figsize = (8,8))
#柱标
labels = 'Frogs','Hogs','Dogs','Logs'
#大小
sizes =[15,30,45,10]
#颜色
colors = ['yellowgreen','gold','lightskyblue','lightcoral']
explode = (0,0.1,0,0)                                           #第二个块分离出来
#绘制饼图
plt.pie(sizes,explode = explode,labels = labels,colors = colors,autopct='%1.1f%%',shadow = True,startangle = 90)
plt.axis('equal')


