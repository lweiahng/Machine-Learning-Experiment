from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

X, y = load_iris(return_X_y=True)
# plt.scatter(X[:,0], X[:, 1],c=y)
# plt.grid()
# plt.show()

cov_mat = np.cov(X.T)
print("协方差矩阵:\n",cov_mat)
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)  # 特征值，特征向量
vec = eig_vec_cov[:, :2]
print("特征向量:\n",vec)
X_new = np.dot(X, vec)  # 矩阵相乘
print(X_new.shape)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.suptitle("电子214刘伟航\n编程矩阵运算实现")
plt.subplot(1,2,1)
plt.title("原样本")
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.grid()

plt.subplot(1,2,2)
plt.title("降维4->2")
plt.scatter(X_new[:, 0], X_new[:, 1], c=y)
plt.grid()
plt.show()


# 4
print(eig_val_cov)
print(eig_val_cov / eig_val_cov.sum())
x = eig_val_cov / eig_val_cov.sum()
plt.figure()
plt.plot(range(1, 5), np.cumsum(x))
plt.grid()
plt.title("电子214刘伟航213876\n信息占比随特征值个数变化")
plt.show()

