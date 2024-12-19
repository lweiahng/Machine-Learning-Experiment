from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.suptitle("电子214刘伟航\nsklearn库实现")

plt.subplot(1,2,1)
plt.title("原样本")
X, y = load_iris(return_X_y=True)
pca=PCA(n_components=2)           # n_components主成分的个数
X_new_sk=pca.fit_transform(X)
print("特征向量:\n",pca.components_)     # 查看特征向量
plt.scatter(X[:,0], X[:, 1],c=y)
plt.grid()

plt.subplot(1,2,2)
plt.title("降维4->2")
plt.scatter(X_new_sk[:,0], X_new_sk[:, 1],c=y)
plt.grid()

plt.show()