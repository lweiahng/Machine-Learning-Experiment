import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.svm import SVC
# 以下为创建数据X 和标签y
np.random.seed(1)
# 创建100 个二维数组，即100 个2 个特征的样本
X = np.random.randn(100, 2)
# np.logical_xor(bool1, bool2)，异或逻辑运算，如果bool1 和bool2 的结果相同则为False，否则为True
# ++和--为一三象限，+-和-+为二四象限，如此做则100 个样本必定线性不可分
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
# 对X 数组人为分类,二四象限为True，即为1 类；一三象限为False，即为-1 类
y = np.where(y, 1, -1)
# 构建决策边界,通用程序，后面可直接使用
def plot_decision_regions(X, y, classifier=None):
    marker_list = ['o', 'x', 's']
    color_list = ['r', 'b', 'g']
    cmap = ListedColormap(color_list[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    t1 = np.linspace(x1_min, x1_max, 666)
    t2 = np.linspace(x2_min, x2_max, 666)
    x1, x2 = np.meshgrid(t1, t2)
    y_hat = classifier.predict(np.array([x1.ravel(), x2.ravel()]).T)
    y_hat = y_hat.reshape(x1.shape)
    plt.contourf(x1, x2, y_hat, alpha=0.2, cmap=cmap)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    for ind, clas in enumerate(np.unique(y)):
        plt.scatter(X[y == clas, 0], X[y == clas, 1], alpha=0.8, s=50,
            c=color_list[ind], marker=marker_list[ind], label=clas)

# 以下为关键语句
# for kernel in ['rbf','linear','poly','sigmoid']:
kernel='rbf'
# gamma = 'auto'
for gamma in[10,50,100]:
    svm = SVC(kernel=kernel, gamma=gamma, C=1, random_state=1)
    svm.fit(X, y)
    plot_decision_regions(X, y, classifier=svm)
    SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',max_iter=-1, probability=False, random_state=1, shrinking=True,tol=0.001, verbose=False)
    plt.title('kernel=%s, gamma=%s,%s'%(kernel,gamma,'电子214刘伟航213876'))
    plt.legend()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from sklearn import datasets
# from sklearn.svm import SVC
from sklearn.datasets import load_iris
# 导入鸢尾花数据集
iris = load_iris()
# 以下为创建数据X 和标签y
X = iris.data[:, 2:4]
y = iris.target[:]
# def plot_decision_regions(X, y, classifier=None):
#     marker_list = ['o', 'x', 's']
#     color_list = ['r', 'b', 'g']
#     cmap = ListedColormap(color_list[:len(np.unique(y))])
#     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     t1 = np.linspace(x1_min, x1_max, 666)
#     t2 = np.linspace(x2_min, x2_max, 666)
#     x1, x2 = np.meshgrid(t1, t2)
#     y_hat = classifier.predict(np.array([x1.ravel(), x2.ravel()]).T)
#     y_hat = y_hat.reshape(x1.shape)
#     plt.contourf(x1, x2, y_hat, alpha=0.2, cmap=cmap)
#     plt.xlim(x1_min, x1_max)
#     plt.ylim(x2_min, x2_max)
#     for ind, clas in enumerate(np.unique(y)):
#         plt.scatter(X[y == clas, 0], X[y == clas, 1], alpha=0.8, s=50,
#             c=color_list[ind], marker=marker_list[ind], label=clas)
#
# # 以下为关键语句
# # for kernel in ['rbf','linear','poly','sigmoid']:
# kernel='rbf'
#     # gamma = 'auto'
# for gamma in[10,50,100]:
#     svm = SVC(kernel=kernel, gamma=gamma, C=1, random_state=1)
#     svm.fit(X, y)
#     plot_decision_regions(X, y, classifier=svm)
#     SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',max_iter=-1, probability=False, random_state=1, shrinking=True,tol=0.001, verbose=False)
#     plt.title('kernel=%s, gamma=%s,%s'%(kernel,gamma,'电子214刘伟航213876'))
#     plt.legend()
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.show()
