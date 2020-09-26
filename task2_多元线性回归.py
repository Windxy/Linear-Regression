from init import *

"""
类名：GradientDescent_MultiLine
功能：实现多元线性回归的梯度下降方法
"""


class GradientDescent_MultiLine:
    def __init__(self, lr, epochs):
        self.lr = lr  # 学习率，用来控制步长（权重调整幅度）
        self.epochs = epochs  # 循环迭代的次数
        self.lose = []  # 损失值计算（损失函数）：均方误差

    '''根据提供的训练数据对模型进行训练'''

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        y = np.squeeze(y)  # 去掉冗余的维度

        self.w = np.zeros(1 + x.shape[1])  # 初始权重，权重向量初始值为0（或任何其他值），长度比X的特征数量多1（多出来的为截距）

        # 开始训练
        for i in range(self.epochs):
            y_hat = np.dot(x, self.w[1:]) + self.w[0]  # 计算预测值
            error = y - y_hat  # 计算真实值与预测值之间的差距
            self.lose.append(np.sum(error ** 2) / 2)  # 将损失加入到损失列表中
            print("迭代次数:{0},进度：{1}%".format(i + 1, 100.0 * (i + 1) / self.epochs), "  loss:", np.sum(error ** 2) / 2)
            # j <- j + α * sum((y - y_hat) * x(j))
            self.w[0] += self.lr * np.sum(error)
            self.w[1:] += self.lr * np.dot(x.T, error)

    '''样本进行预测'''

    def predict(self, x):
        x = np.asarray(x)  # 测试样本
        result = np.dot(x, self.w[1:]) + self.w[0]  # 预测结果
        return result

    '''损失收敛图'''

    def loss_ans_show(self):
        if len(self.lose) == 0:
            print("你还没有训练噢")
            return
        x = range(len(self.lose))
        plt.plot(x, self.lose, 'r', label='损失函数')
        plt.title('loss收敛图')
        plt.ylabel("loss")
        plt.xlabel("iter")
        plt.legend()
        plt.show()


'''   
函数名：lms(least square method)
功能：实现多元线性回归的公式法
'''


def lms(x_train, x_test, y_train, y_test):
    theta_n = np.dot(np.dot(inv(np.dot(x_train.T, x_train)), x_train.T), y_train)  # theta = (X`X)^(-1)X`Y，其中X`表示X的转置
    y_pre = np.dot(x_test, theta_n)
    mse = np.average((y_test - y_pre) ** 2)
    return theta_n, y_pre, mse


"""
函数名：fit_transform
功能：对数据进行标准化处理,即均值为0，标准差为1
"""


def fit_transform(x):
    x = np.asarray(x)
    std_ = np.std(x, axis=0)  # 标准差
    mean_ = np.mean(x, axis=0)  # 均值
    return (x - mean_) / std_


if __name__ == '__main__':

    '''载入数据'''
    data = pd.read_csv("housing.csv")
    head = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAS', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    # plt.scatter(range(len(data[[head[0]]])),data[[head[0]]])
    fig = plt.figure()
    for i in range(14):
        plt.subplot(7, 2, i + 1)
        # sns.set_style('whitegrid')
        # sns.distplot(data[head[i]], kde=True, rug=True,cumulative=True,
        #              rug_kws={'color': 'y', 'lw': 2, 'alpha': 0.5, 'height': 0.1},  # 设置数据频率分布颜色#控制是否显示观测的小细条（边际毛毯）
        #              kde_kws={"color": "y", "lw": 1.5, 'linestyle': '--'},  # 设置密度曲线颜色，线宽，标注、线形，#控制是否显示核密度估计图
        #              label=[head[i]])
        '''核密度估计，可以比较直观的看出数据样本本身的分布特征'''
        sns.kdeplot(data[head[i]], cumulative=False, bw=1.5, label=head[i],
                    color='r', lw=2, alpha=0.5, shade=True)
    plt.show()
    print(data.head())

    '''特征缩放,构造数据'''
    data_s = fit_transform(data)
    x = data_s[:, :-1]
    y = data_s[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

    for i in range(14):
        plt.subplot(7, 2, i + 1)
        sns.set_style('whitegrid')
        # sns.distplot(data[head[i]], kde=True, rug=True,cumulative=True,
        #              rug_kws={'color': 'y', 'lw': 2, 'alpha': 0.5, 'height': 0.1},  # 设置数据频率分布颜色#控制是否显示观测的小细条（边际毛毯）
        #              kde_kws={"color": "y", "lw": 1.5, 'linestyle': '--'},  # 设置密度曲线颜色，线宽，标注、线形，#控制是否显示核密度估计图
        #              label=[head[i]])
        ser = pd.Series(data_s[:, i])
        sns.kdeplot(ser, cumulative=False, bw=1.5, label=head[i],
                    color='r', lw=2, alpha=0.5, shade=True)
    plt.show()

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题

    '''1.梯度下降法'''
    if True:
        GD = GradientDescent_MultiLine(lr=0.0001, epochs=1000)
        GD.fit(x_train, y_train)
        pre_gd = GD.predict(x_test)
        mse_gd = np.average((y_test - pre_gd) ** 2)
        GD.loss_ans_show()

    '''2.公式法：least square method'''
    if True:
        theta_n, pre_lms, mse_lms = lms(x_train, x_test, y_train, y_test)

    '''3.sklearn'''
    if True:
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)
        pre_sk = regr.predict(x_test)
        mse_sk = np.average((y_test - pre_sk) ** 2)

    '''模型评价'''
    print('梯度下降法：{}%'.format((1 - mse_gd) * 100))
    print('公式法：{}%'.format((1 - mse_lms) * 100))
    print('sklearn法：{}%'.format((1 - mse_sk) * 100))

    '''可视化'''
    plt.figure(figsize=(10, 10))
    plt.plot(pre_gd, label="梯度下降")
    plt.plot(pre_lms, label="公式法")
    plt.plot(pre_sk, label="sklearn")  # 可以发现sklearn也是公式法
    plt.plot(y_test, label="真实值")
    plt.title("线性回归预测-梯度下降法")
    plt.xlabel("样本序号")
    plt.ylabel("房价")
    plt.legend()
    plt.show()
