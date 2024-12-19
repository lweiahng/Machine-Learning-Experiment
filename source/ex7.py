import numpy as np
import matplotlib.pyplot as plt
# 样本点
x=np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y=np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])
# 计算样本均值
mean_x=np.mean(x)
mean_y=np.mean(y)
# 样本去均值
scaled_x=x-mean_x
scaled_y=y-mean_y
# 样本点矩阵化
data=np.matrix([[scaled_x[i],scaled_y[i]] for i in range(len(scaled_x))])

# 一行两列子图，第一幅绘制样本点，第二幅绘制去均值的样本点
fig=plt.figure()
fig.suptitle("电子214刘伟航213876")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(121)
plt.title("原样本")
plt.plot(x,y,'d')
plt.subplot(122)
plt.title("去均值化")
plt.plot(scaled_x,scaled_y,'o')
plt.show()

# 求协方差矩阵
cov=np.cov(scaled_x,scaled_y)
print('cov=',cov)
# 计算特征值、特征向量；输出
eig_val, eig_vec = np.linalg.eig(cov)
print('eig_val=',eig_val)
print('eig_vec=',eig_vec)

# # 根据样本点范围设置绘图区域X，Y坐标范围
# xmin ,xmax = scaled_x.min(), scaled_x.max()
# ymin, ymax = scaled_y.min(), scaled_y.max()
# dx = (xmax - xmin) * 0.2
# dy = (ymax - ymin) * 0.2
# plt.xlim(xmin - dx, xmax + dx)
# plt.ylim(ymin - dy, ymax + dy)

# 特征向量与样本点做矩阵乘法，进行降维，转置
new_data=np.transpose(np.dot(eig_vec,np.transpose(data)))
# 分别绘制特征向量矩阵的两列向量的方向
plt.axis('equal')
plt.title("电子214刘伟航213876\n特征向量指示的两个主成分方向")
plt.plot([eig_vec[:,0][0],0],[eig_vec[:,0][1],0],color='red')
plt.plot([eig_vec[:,1][0],0],[eig_vec[:,1][1],0],color='blue')
plt.show()

# 改变数据维度
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
# 根据特征值从大到小，对特征向量排序
eig_pairs.sort(reverse=True)
# 取出特征值最大对应的特征向量feature
feature=eig_pairs[0][1]
# 降到一维
new_data_reduced=np.transpose(np.dot(feature,np.transpose(data)))
print('new_data_reduced = ')
print(new_data_reduced)

# 绘制去均值的样本点
p1, =plt.plot(scaled_x,scaled_y,'o',color='red')
# 绘制降维的特征向量方向
p2, =plt.plot([eig_vec[:,0][0],0],[eig_vec[:,0][1],0],color='red')
p3, =plt.plot([eig_vec[:,1][0],0],[eig_vec[:,1][1],0],color='blue')
# 绘制降维后的数据点
p4, =plt.plot(new_data[:,0],new_data[:,1],'^',color='blue')
p5, =plt.plot(new_data_reduced[:,0],[1.2]*10,'*',color='green')
plt.legend([p1,p2,p3,p4,p5],["去均值的样本点","特征向量方向","特征向量方向","按特征值方向经过旋转的样本点","降维后的样本分布"])
# plt.plot(new_data_reduced[:,0]*feature[0],new_data_reduced[:,0]*feature[1],'.',color='maroon')
plt.axis('equal')
plt.title("电子214刘伟航213876")
plt.show()
