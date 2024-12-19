#导包
import numpy as np
import pandas  as pd
from pandas import DataFrame,Series

#手动创建训练数据集
feature = np.array([[170,65,41],[166,55,38],[177,80,39],[179,80,43],[170,60,40],[170,60,38]])
target = np.array(['男','女','女','男','女','女'])

from sklearn.neighbors import KNeighborsClassifier #k邻近算法模型

#实例k邻近模型，指定k值=3
knn = KNeighborsClassifier(n_neighbors=3)

#训练数据
knn.fit(feature,target)

#模型评分
knn.score(feature,target)

#预测
print(knn.predict(np.array([[176,71,38]])))