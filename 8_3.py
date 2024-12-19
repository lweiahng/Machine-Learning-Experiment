# 色泽{青绿：1，乌黑：2，浅白：3}
# 根蒂{蜷缩：1，稍蜷：2，硬挺：3}
# 敲声{浊响：1，沉闷：2，清脆：3}
# 纹理{清晰：1，稍糊：2，模糊：3}
# 脐部{凹陷：1，稍凹：2，平坦：3}
# 触感{硬滑：1，软粘：2}
# 好瓜{是：1，否：0}
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import pydotplus

melon_feature_E = 'Color', 'Root', 'Knocking', 'Texture', 'Navel', 'Touch'
melon_class = 'yes', 'no'

# 1、读入数据，并将原始数据中的数据转换为数字形式
data = np.loadtxt("watermelon2_0.txt", delimiter="\t", dtype=str)
# 第0列用out—look_type操作。
print(data)

x, y = np.split(data, (6,), axis=1)

# 2、拆分训练数据与测试数据，为了进行交叉验证
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# 3、使用信息熵作为划分标准，对决策树进行训练
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
clf.fit(x_train, y_train)

# 4、把决策树结构写入文件
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=melon_feature_E, class_names=melon_class,
                                filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('melon.pdf')

# 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大
print(clf.feature_importances_)

# 5、使用训练数据预测，预测结果完全正确
answer = clf.predict(x_train)
y_train = y_train.reshape(-1)
print(answer)
print(y_train)
print(np.mean(answer == y_train))

# 6、对测试数据进行预测，准确度较低，说明过拟合
answer = clf.predict(x_test)
y_test = y_test.reshape(-1)
print(answer)
print(y_test)
print(np.mean(answer == y_test))