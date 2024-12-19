import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.neighbors import KNeighborsClassifier  # k邻近算法模型
import matplotlib.pyplot as plt

x_train = np.array(
    [[2, 1], [3, 2], [4, 2], [1, 3], [1.5, 4], [1.7, 3], [2.6, 5], [3.4, 3], [3, 6], [1, 7], [4, 5], [1.2, 6], [1.8, 7],
     [2.2, 8], [3.7, 7], [4.8, 5]])
y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
x_test = np.array([[2, 5]])  # X_test 为测试样本

X_train_0 = np.array([x_train[i, :] for i in range(len(y_train)) if y_train[i] == 0])
# 将class1 一类的样本点放到X_train_1 中
X_train_1 = np.array([x_train[i, :] for i in range(len(y_train)) if y_train[i] == 1])
# 绘制所有样本点并采用不同的颜色分别标记class0 以及class1
fig = plt.figure()
fig.suptitle("电子214刘伟航213876, test:{}".format(x_test))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
for k in [1, 3, 5, 7]:  # k 为邻居数
    plt.subplot(2, 2, (k + 1) // 2)
    plt.title("k=%s" % (k))

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    knn.score(x_train, y_train)
    y_test = knn.predict(x_test)
    y_c = 'g' if y_test == 0 else 'm'
    plt.scatter(X_train_0[:, 0], X_train_0[:, 1], c='g', marker='o', label='train_class0')
    plt.scatter(X_train_1[:, 0], X_train_1[:, 1], c='m', marker='o', label='train_class1')
    plt.scatter(x_test[0][0], x_test[0][1], c=y_c, marker='*', s=100, label='test_class')
    print(x_train, y_train, x_test)
    print('预测分类：', knn.predict(x_test))
    plt.legend(loc='best')
plt.show()
