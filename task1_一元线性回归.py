from init import *

'''
类名：GeadientDescent_Line
功能：实现一元线性回归的梯度下降方法
'''
class GradientDescent_Line():
    def __init__(self, DataFile="data.csv", AutoLearn = True, lr=0.00001, b=0, w=0, epochs=50,):
        '''
        参数的设置
        '''
        '''参数设置'''
        self.lr = lr                # 学习率learning rate
        self.AutoLearn = AutoLearn  # 是否自适应调整学习率
        self.b = b                  # 初始截距
        self.w = w                  # 初始斜率
        self.epochs = epochs        # 最大迭代次数

        '''载入数据'''
        data = pd.read_csv(DataFile)#数据
        t = data.sample(len(data), random_state=0)      # 随机抽样，打乱顺序
        self.X = t['x']
        self.Y = t['y']
        self.lens = len(self.X)     #数据总数

    '''Loss损失函数'''
    def compute_error(self, b, w):
        error_all = 0       # 损失和
        for i in range(self.lens):
            error_all += (self.Y[i] - (w * self.X[i] + b)) ** 2
        return error_all / (self.lens * 2.0)

    '''梯度下降 GD'''
    def fit(self):
        # 定义相关变量
        flag = 0
        last = 0
        loss = []

        # 动量学习参数
        a = 0.5
        v1 = 0.1
        v2 = 0.1

        for i in range(self.epochs):
            b_grad = 0
            w_grad = 0
            # 计算梯度的总和再求平均
            for j in range(self.lens):
                b_grad += (1.0/self.lens) * (((self.w * self.X[j]) + self.b) - self.Y[j])
                w_grad += (1.0/self.lens) * self.X[j] * (((self.w * self.X[j]) + self.b) - self.Y[j])
            v1 = a * v1 - self.lr * b_grad
            v2 = a * v2 - self.lr * w_grad
            self.b = self.b + v1
            self.w = self.w + v2
            now = self.compute_error(self.b, self.w)
            print("迭代次数:{0},进度：{1}%".format(i+1,100.0*(i+1)/self.epochs),"  loss:",now)

            '''自动学习率调整'''
            if self.AutoLearn:
                if last < now:
                    last = now
                    flag+=1
                if flag==3:
                    self.lr*=0.5
                    flag = 0
                    print("调整学习率:",self.lr)

            loss.append(now)

        return self.b, self.w, loss

    '''源数据图'''
    def line_show(self):
        plt.ylabel("y")
        plt.xlabel("x")
        plt.plot(self.X, self.Y, 'b.',label='源数据')
        plt.title("源数据分布图")
        plt.legend()
        plt.show()

    '''回归结果图'''
    def get_regress_ans_show(self,b, w):
        plt.ylabel("y")
        plt.xlabel("x")
        plt.plot(self.X, self.Y, 'b.',label='源数据')
        plt.plot(self.X, w*self.X + b, 'r',label='回归线')
        plt.title("回归结果图")
        plt.legend()
        plt.show()

    '''loss收敛图'''
    def loss_ans_show(self,loss):
        x = range(len(loss))
        plt.plot(x, loss, 'r', label='损失函数')
        plt.title('loss收敛图')
        plt.ylabel("loss")
        plt.xlabel("iter")
        plt.legend()
        plt.show()

    '''3维图'''
    def loss3D(self,X,Y,Z,surface=True):
        from matplotlib.colors import LightSource
        fig = plt.figure()
        ax = Axes3D(fig)
        if surface:
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)
        # ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
        ax.contour(X, Y, Z, cmap=cm.coolwarm)
        # ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        plt.show()

'''
函数名：sklearn_Line
功能：实现一元线性回归的sklearn方法
'''
def sklearn_Line(DataFile="data.csv"):
    # 载入数据
    data = np.genfromtxt(DataFile, delimiter=",")
    x_data = data[1:, 1]
    y_data = data[1:, 2]
    # plt.scatter(x_data, y_data)
    # plt.show()
    print(x_data.shape)

    x_data = x_data[:, np.newaxis]
    y_data = y_data[:, np.newaxis]

    # 创建并拟合模型
    model = linear_model.LinearRegression()
    model.fit(x_data, y_data)

    # 画图
    plt.scatter(x_data, y_data, c='b', s=4, label='源数据')
    plt.plot(x_data, model.predict(x_data), 'r', label='回归线')
    plt.title("sklearn方法_回归结果图")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    '''梯度下降法'''
    if True:
        '''载入数据'''
        datafile = 'data.csv'
        '''定义梯度下降类'''
        GD_Line = GradientDescent_Line(DataFile=datafile)
        '''进行梯度下降'''
        b, w, loss = GD_Line.fit()
        print("训练完成")

        '''可视化'''
        GD_Line.line_show()
        GD_Line.get_regress_ans_show(b, w)
        GD_Line.loss_ans_show(loss)

        '''可视化三维关系图'''
        w_show = np.arange(0, 5, 0.1)
        b_show = np.arange(-1000, 1000, 100)
        w_show,b_show = np.meshgrid(w_show, b_show)
        loss_show = GD_Line.compute_error(b_show,w_show)
        GD_Line.loss3D(w_show,b_show,loss_show)
        GD_Line.loss3D(w_show,b_show,loss_show,surface=False)

    '''调用sklearn法'''
    if True:
        sklearn_Line("data.csv")