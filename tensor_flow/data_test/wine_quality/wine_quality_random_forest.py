import wine_quality_data
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

(train_x, train_y), (test_x, test_y) = wine_quality_data.load_data_split_by_sample()

# clf = SVC(kernel='rbf', C=10000)
# clf.fit(train_x, train_y)
# res = clf.predict(test_x)

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(train_x, train_y)
pred_rfc = rfc.predict(test_x)

print(accuracy_score(test_y, pred_rfc))


clf = SVC(kernel='rbf', C=10000)
clf.fit(train_x, train_y)
res = clf.predict(test_x)
print(accuracy_score(test_y, res))

model = SVC(kernel='linear', C=0.1, gamma=0.1)
model.fit(train_x, train_y)
prediction2 = model.predict(test_x)
print('Accuracy for linear SVM is', accuracy_score(prediction2, test_y))


# clf.fit(features_train, labels_train)
# res = clf.predict(features_test)

