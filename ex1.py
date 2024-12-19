# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.rcParams['axes.labelsize'] = 14
# plt.rcParams['xtick.labelsize'] = 12
# plt.rcParams['ytick.labelsize'] = 12
#
# X = 2 * np.random.rand(100, 1)
# y = 3*X + 4 + np.random.randn(100, 1)
# # 构造线性方程，加入随机抖动
# X_b = np.c_[(np.ones((100, 1)), X)]
# # np.linalg.inv:矩阵求逆
# theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# print(theta_best)
#
# # 测试数据
# X_new = np.array([[0], [2]])
# X_new_b = np.c_[np.ones((2, 1)), X_new]
# # 预测结果
# y_predict = X_new_b.dot(theta_best)
# print(y_predict)
#
# plt.plot(X, y, 'b.')     # b指定为蓝色,.指定线条格式
# plt.xlabel('X_1')
# plt.ylabel('y')
# plt.axis([0, 2, 0, 15])
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.title('电子214-刘伟航-213876')
# plt.plot(X_new, y_predict, 'r--o')   # 指定红色和线条
# plt.plot(X, y, 'b.')     # 指定蓝色和点
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# x=np.array([1,2,3,4,5])
# y=np.array([1,3,2,3,5])
# p,q=0,0
# ax=sum(x)/len(x)
# ay=sum(y)/len(y)
# for i in range(len(x)):
#     p+=(x[i]-ax)*(y[i]-ay)
#     q+=(x[i]-ax)**2
# b0=p/q
# b1=ay-b0*ax
# xx=np.array([min(x),max(x),2.5,4.5])
# yy=xx.dot(b0)+b1
# plt.plot(x,y,'b.')
# plt.plot(xx[:2],yy[:2],'r--')
# plt.plot(xx[2:],yy[2:],'ro')
# plt.axis([0.5, 5.5, 0, 6])
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.title('电子214-刘伟航-213876')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

# 导入鸢尾花数据集
iris = load_iris()
X = iris.data[100:150, 2:4]  ##鸢尾花数据集后50个样本,只取特征空间中后两个维度
print(len(X))
print(X)
x1=X[0:50, 0]  #第一列数据
print(x1.shape) # 第一列大小为50行,1列
x2=X[0:50, 1]  #第二列数据
print(x2.shape) # 第二列大小为50行,1列
plt.scatter(x1, x2, c="red", marker='o', label='class3')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=1)   #图例放置位置,右上为1,逆时针排序
plt.rcParams['font.sans-serif'] = ['SimHei']
X_b = np.c_[(np.ones((50, 1)), x1)]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(x2)
print(theta_best)
X_new = np.array([[4], [8]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
print(y_predict)
plt.plot(X_new, y_predict, 'b--')
plt.title(['电子214-刘伟航-213876','X2='+str(round(theta_best[1],2))+'*X1+'+str(round(theta_best[0],2))])
plt.show()