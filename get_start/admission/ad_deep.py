import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

from sklearn.metrics import r2_score

df = pd.read_csv("data/Admission_Predict.csv", sep=",")
df.info()
df = shuffle(df)
df = df.rename(columns={'Chance of Admit ':'Chance of Admit'})
df.drop(["Serial No."], axis=1, inplace=True)
y = df["Chance of Admit"].values
# x = df.drop(["Chance of Admit"],axis=1)
x_data = df.drop(["Chance of Admit"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

x_train, x_test,y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

model = Sequential()
# model.add(Dense(1, input_dim=7))

model.add(Dense(300, input_dim=7, kernel_initializer='normal', activation='relu'))
model.add(Dense(200, activation='relu', bias_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

history = model.fit(x_train, y_train, epochs=100, batch_size=20,  verbose=1, validation_split=0.2)



print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(model.predict(x_test.iloc[[1],:])))

y_head_lr = model.predict(x_test)

print("r_square score ttttt: ", r2_score(y_train,model.predict(x_train)))
print("r_square score: ", r2_score(y_test,y_head_lr))


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_head_lr = lr.predict(x_test)
print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(lr.predict(x_test.iloc[[1],:])))
print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(lr.predict(x_test.iloc[[2],:])))
from sklearn.metrics import r2_score
print("r_square score: ", r2_score(y_test,y_head_lr))

# print(history.history.keys())
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
