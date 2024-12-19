import numpy as np
import matplotlib.pyplot as plt
# 初始化模拟数据， X_train 为样本点
X_train = np.array([[2, 1],[3, 2],[4, 2],[1, 3],[1.5, 4],[1.7, 3],[2.6, 5],[3.4, 3],[3, 6],[1, 7],[4, 5],[1.2, 6],[1.8, 7],[2.2, 8],[3.7, 7],[4.8, 5]])
y_train = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]) # y_train 为样本点标记
X_test = np.array([2,5]) # X_test 为测试样本
fig=plt.figure()
fig.suptitle("电子214刘伟航213876, test:{}".format(X_test))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
for k in [1,3,5,7]: # k 为邻居数
    plt.subplot(2,2,(k+1)//2)
    plt.title("k=%s"%(k))
    # 这里的距离公式采用欧式距离
    square_ = (X_train - X_test) ** 2
    square_sum = square_.sum(axis=1) ** 0.5
    # 根据距离大小排序并找到测试样本与所有样本k 个最近的样本的序列
    square_sum_sort = square_sum.argsort()
    small_k = square_sum_sort[:k]
    # K 近邻用于分类则统计K 个邻居分别属于哪类的个数，用于回归则计算K 个邻居的y 的平均值作为预测结果
    # 统计距离最近的k 个样本分别属于哪一类的个数别返回个数最多一类的序列作为预测结果
    y_test_sum = np.bincount(np.array([y_train[i] for i in small_k])).argsort()
    # 打印预测结果
    print('predict: class {}'.format(y_test_sum[-1]))
    # 将数据可视化更生动形象
    # 将class0 一类的样本点放到X_train_0 中
    X_train_0 = np.array([X_train[i, :] for i in range(len(y_train)) if y_train[i] == 0])
    # 将class1 一类的样本点放到X_train_1 中
    X_train_1 = np.array([X_train[i, :] for i in range(len(y_train)) if y_train[i] == 1])
    # 绘制所有样本点并采用不同的颜色分别标记class0 以及class1
    plt.scatter(X_train_0[:,0], X_train_0[:,1], c='g', marker='o', label='train_class0')
    plt.scatter(X_train_1[:,0], X_train_1[:,1], c='m', marker='o', label='train_class1')
    if y_test_sum[-1] == 0:
        test_class = 'g'
    elif y_test_sum[-1] == 1:
        test_class = 'm'
    plt.scatter(X_test[0], X_test[1], c=test_class, marker='*', s=100, label='test_class')
    # 连接测试样本与k 个近邻
    for i in small_k:
        plt.plot([X_test[0], X_train[i, :][0]], [X_test[1], X_train[i, :][1]], c='c')
    plt.legend(loc='best')
plt.show()
