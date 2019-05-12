import wine_quality_data
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVC
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
import xgboost as xgb


(train_x, train_y), (test_x, test_y) = wine_quality_data.load_data_split_by_sample()


clf = SVC(kernel='rbf', C=10000)
clf.fit(train_x, train_y)
res = clf.predict(test_x)
print('Accuracy for SVM is', accuracy_score(test_y, res))

model = SVC(kernel='linear', C=0.1, gamma=0.1)
model.fit(train_x, train_y)
prediction2 = model.predict(test_x)
print('Accuracy for linear SVM is', accuracy_score(prediction2, test_y))


model = LogisticRegression()
model.fit(train_x, train_y)
prediction3 = model.predict(test_x)
print('The accuracy of the Logistic Regression is',accuracy_score(prediction3, test_y))


model = DecisionTreeClassifier()
model.fit(train_x, train_y)
prediction4=model.predict(test_x)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction4,test_y))


model = KNeighborsClassifier()
model.fit(train_x, train_y)
prediction5=model.predict(test_x)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction5,test_y))


# model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
# model.fit(train_x, train_y)
#
#     # 对测试集进行预测
# ans = model.predict(test_y)
# print('The accuracy of the KNN is',metrics.accuracy_score(ans,test_y))


# clf.fit(features_train, labels_train)
# res = clf.predict(features_test)

