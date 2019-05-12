from sklearn import datasets
#导入SVM模型
from sklearn import svm

from sklearn.externals import joblib



#加载数据集
iris=datasets.load_iris()
# 查看数据集大小
print(iris.data.shape)


#建立线性SVM模型
clf=svm.LinearSVC()
#用数据训练模型
clf.fit(iris.data,iris.target)
#训练好模型后，用新数据进行预测
clf.predict([[5.0,3.6,1.3,0.25]])
#查看训练好的模型的参数
clf.coef_



from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(iris.data, iris.target)
print(clf.predict([[5.0, 3.6, 1.3, 0.25]]))



joblib.dump(clf, '/Users/apple/Desktop/model.pkl')
model = joblib.load('/Users/apple/Desktop/model.pkl')
model.predict([[5.0,3.6,1.3,0.25]])
