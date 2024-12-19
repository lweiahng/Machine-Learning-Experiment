from sklearn import datasets
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import copy



def find_neighbor(j, x, eps):
    N = list()
    for i in range(x.shape[0]):
        temp = np.sqrt(np.sum(np.square(x[j] - x[i])))  # 计算欧式距离
        if temp <= eps:
            N.append(i)
    return set(N)


def DBSCAN(X, eps, min_Pts):
    k = -1
    neighbor_list = []  # 用来保存每个数据的邻域
    omega_list = []  # 核心对象集合
    gama = set([x for x in range(len(X))])  # 初始时将所有点标记为未访问
    cluster = [-1 for _ in range(len(X))]  # 聚类
    for i in range(len(X)):
        neighbor_list.append(find_neighbor(i, X, eps))
        if len(neighbor_list[-1]) >= min_Pts:
            omega_list.append(i)  # 将样本加入核心对象集合
    omega_list = set(omega_list)  # 转化为集合便于操作
    while len(omega_list) > 0:
        gama_old = copy.deepcopy(gama)
        j = random.choice(list(omega_list))  # 随机选取一个核心对象
        k = k + 1
        Q = list()
        Q.append(j)
        gama.remove(j)
        while len(Q) > 0:
            q = Q[0]
            Q.remove(q)
            if len(neighbor_list[q]) >= min_Pts:
                delta = neighbor_list[q] & gama
                deltalist = list(delta)
                for i in range(len(delta)):
                    Q.append(deltalist[i])
                    gama = gama - delta
        Ck = gama_old - gama
        Cklist = list(Ck)
        for i in range(len(Ck)):
            cluster[Cklist[i]] = k
        omega_list = omega_list - Ck
    return cluster


X1, y1 = datasets.make_circles(n_samples=2000, factor=.6, noise=.02)
X2, y2 = datasets.make_blobs(n_samples=400, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[.1]], random_state=9)
X = np.concatenate((X1, X2))
eps = 0.08
min_Pts = 10
begin = time.time()
C = DBSCAN(X, eps, min_Pts)
end = time.time()
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=C)
plt.title('电子214刘伟航213876')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.show()

n_samples=200
for i in range(3):
    x=0.2*i
    for j in range(3):
        cluster_std = j
        # X1, y1 = datasets.make_circles(n_samples=n_samples, factor=factor, noise=noise)
        X2, y2 = datasets.make_blobs(n_samples=n_samples, n_features=2, centers=[[x, x]], cluster_std=[[cluster_std]], random_state=9)
        plt.subplot(3,3,i*3+j+1)
        # plt.scatter(X1[:,0],X1[:,1])
        plt.scatter(X2[:,0],X2[:,1])
        plt.title('center:x=%.2f,y=%.2f,std=%.2f'%(x,x,cluster_std))
        plt.gca().set_aspect(1)
plt.show()

def find_neighbor(j, x, eps):
    N = list()
    for i in range(x.shape[0]):
        temp = np.sqrt(np.sum(np.square(x[j] - x[i])))  # 计算欧式距离
        if temp <= eps:
            N.append(i)
    return set(N)


def DBSCAN(X, eps, min_Pts):
    k = -1
    neighbor_list = []  # 用来保存每个数据的邻域
    omega_list = []  # 核心对象集合
    gama = set([x for x in range(len(X))])  # 初始时将所有点标记为未访问
    cluster = [-1 for _ in range(len(X))]  # 聚类
    for i in range(len(X)):
        neighbor_list.append(find_neighbor(i, X, eps))
        if len(neighbor_list[-1]) >= min_Pts:
            omega_list.append(i)  # 将样本加入核心对象集合
    omega_list = set(omega_list)  # 转化为集合便于操作
    while len(omega_list) > 0:
        gama_old = copy.deepcopy(gama)
        j = random.choice(list(omega_list))  # 随机选取一个核心对象
        k = k + 1
        Q = list()
        Q.append(j)
        gama.remove(j)
        while len(Q) > 0:
            q = Q[0]
            Q.remove(q)
            if len(neighbor_list[q]) >= min_Pts:
                delta = neighbor_list[q] & gama
                deltalist = list(delta)
                for i in range(len(delta)):
                    Q.append(deltalist[i])
                    gama = gama - delta
        Ck = gama_old - gama
        Cklist = list(Ck)
        for i in range(len(Ck)):
            cluster[Cklist[i]] = k
        omega_list = omega_list - Ck
    return cluster


# X1, y1 = datasets.make_moons(n_samples=2000, shuffle=True, noise=0.1, random_state=None)
# eps = 0.12
# min_Pts = 10
# begin = time.time()
# C = DBSCAN(X1, eps, min_Pts)
# end = time.time()
# plt.figure()
# plt.scatter(X1[:, 0], X1[:, 1], c=C)
# plt.title('电子214刘伟航213876,eps=%.2f,min_Pts=%.0f'%(eps,min_Pts))
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.show()



X1, y1 = datasets.make_circles(n_samples=500,factor=.95,noise=.015)
X2, y2 = datasets.make_blobs(n_samples=50, n_features=2, centers=[[-0.5, 0.4]], cluster_std=[[.08]], random_state=9)
X3, y3 = datasets.make_blobs(n_samples=50, n_features=2, centers=[[0.5, 0.4]], cluster_std=[[.08]], random_state=9)
X4, y4 = datasets.make_blobs(n_samples=10, n_features=2, centers=[[0, -0.5]], cluster_std=[[.05]], random_state=9)
X5, y5 = datasets.make_blobs(n_samples=8, n_features=2, centers=[[0.1, -0.45]], cluster_std=[[.05]], random_state=9)
X6, y6 = datasets.make_blobs(n_samples=8, n_features=2, centers=[[-0.1, -0.45]], cluster_std=[[.05]], random_state=9)
X7, y7 = datasets.make_blobs(n_samples=5, n_features=2, centers=[[-0.2, -0.4]], cluster_std=[[.05]], random_state=9)
X8, y8 = datasets.make_blobs(n_samples=5, n_features=2, centers=[[0.2, -0.4]], cluster_std=[[.05]], random_state=9)
X = np.concatenate((X1, X2, X3, X4,X5,X6,X7,X8))
eps = 0.12
min_Pts = 8
C = DBSCAN(X, eps, min_Pts)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=C)
# plt.scatter(X[:, 0], X[:, 1])
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('电子214刘伟航213876')
plt.gca().set_aspect(1)
plt.show()
