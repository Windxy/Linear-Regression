import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.model_selection import train_test_split
import matplotlib as mpl
from sklearn import linear_model
from numpy.linalg import inv
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei'] # 替换sans-serif字体
plt.rcParams['axes.unicode_minus'] = False   # 解决坐标轴负数的负号显示问题
