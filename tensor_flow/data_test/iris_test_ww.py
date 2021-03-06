from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

# todo 必要， 但是不知道干啥用的
tf.enable_eager_execution()

data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"






class_dict = dict({'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3})

# # 获取数据路径
path =tf.keras.utils.get_file(fname=os.path.basename(data_url), origin=data_url)
# print("Local copy of the dataset file: {}".format(path))
# # 获取dataset 数据
dataset = tf.data.TextLineDataset(path)

# def label_test(t):
#     if(t=='Iris-setosa'):
#         return 1
#     else:
#         return 0


# 解析 csv
def parse_csv(line):
  example_defaults = [[0.], [0.], [0.], [0.], [""]]  # sets field types
  # c_name=['a', 'b', 'c', 'd', 'e']
  parsed_line = tf.decode_csv(line, example_defaults)

  features = tf.reshape(parsed_line[:-1], shape=(4,))
  # Last field is the label
  label = tf.reshape(parsed_line[-1], shape=())

  # label = tf.string_to_number(label, tf.int64)

  # tf.Print(label)



  # label = tf.cast(label, tf.int8)
  # label = label.map()
  return features, label

train_dataset = dataset.map(parse_csv)      # parse each row



# train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
train_dataset = train_dataset.batch(32)

features, label = iter(train_dataset).next()
print("example features:", features[0])
print("example label:", label[0])


# 创建模型
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(3)
])


def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)



# 优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)








train_loss_results = []
train_accuracy_results = []

# num_epochs = 201
num_epochs = 500

for epoch in range(num_epochs):
  epoch_loss_avg = tfe.metrics.Mean()
  epoch_accuracy = tfe.metrics.Accuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.variables),
                              global_step=tf.train.get_or_create_global_step())

    # Track progress
    epoch_loss_avg(loss(model, x, y))  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))




    # 图表展示
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)

plt.show()



# 获取测试数据



test_url = "http://download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

test_dataset = tf.data.TextLineDataset(test_fp)
test_dataset = test_dataset.skip(1)             # skip header row
test_dataset = test_dataset.map(parse_csv)      # parse each row with the funcition created earlier
test_dataset = test_dataset.shuffle(1000)       # randomize
test_dataset = test_dataset.batch(32)           # use the same batch size as the training set

print(1)

# 准确率


test_accuracy = tfe.metrics.Accuracy()

for (x, y) in test_dataset:
  prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))




# class_ids = ["Iris setosa", "Iris versicolor", "Iris virginica"]

predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5],
    [8.3, 7.0, 5.5, 2.5],
    [5.9, 3.0, 4.2, 1.5],
    [6.9, 3.1, 5.4, 2.1],
    [1.9, 2.1, 3.4, 5.1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits)
  # name = class_ids[class_idx]
  print("Example {} prediction: {}".format(i, class_idx))



print("shape=", dataset.output_shapes)
print("first=", iter(dataset).next())



#
print(dataset)
