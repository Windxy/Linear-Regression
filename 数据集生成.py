# coding:utf-8
'''
**************************************************
@File   ：机器学习 -> 数据集
@IDE    ：PyCharm
@Author ：Small_wind
@Date   ：2020/9/23 9:21
**************************************************
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] # 替换sans-serif字体
plt.rcParams['axes.unicode_minus'] = False   # 解决坐标轴负数的负号显示问题

'''生成数据集'''
def generate_data(w = 0.15, b = 20, nums = 100):
    x = np.arange(nums)#生成1000个数据先
    # ran = np.random.randint(low=-64,high = 64,size=len(x))
    ran = np.random.normal(loc=0, scale=5.0, size=len(x))        # 均值、 方差
    # np.random.seed(0)
    # ran = np.random.randint(100,500,nums)
    y = x*w + b + ran
    return x,y

'''生成一元线性数据可视化图,散点图'''
def show_data(x,y):
    fig = plt.figure()  # 创建一个空窗口
    ax1 = fig.add_subplot(1, 1, 1)  # 1x1个子图中，第一个
    # ax1.scatter(x, y, color='b',s=4)
    plt.plot(x, y, 'b.', label='源数据')
    plt.ylabel("y")
    plt.xlabel("x")
    plt.title(r'源数据分布图')
    plt.legend()
    plt.show()

'''写入csv文件'''
def writre_csv(x,y):
    data = {'x':x,'y':y}
    frame = pd.DataFrame(data,columns=['x','y'])
    frame.to_csv('data2.csv')

x,y = generate_data()
show_data(x,y)
writre_csv(x,y)