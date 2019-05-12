import tensorflow as tf
# 下载或者加载 加载数据集合 start
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 下载或者加载   加载数据集合 end

# 建立占位符，用于存储每个图片的 向量值 ,图片宽高为  所以图片 共有 28*28 =784 个像素点
x = tf.placeholder(tf.float32, [None, 784])

# 创建 784 *10 的权重 矩阵,  因为是 分类 10 个数字 ，所以是   784*10
W = tf.Variable(tf.zeros([784, 10]))

# 创建偏执一维矩阵 即向量。
b = tf.Variable(tf.zeros([10]))

# 利用公式  x*w +b 计算 每张图片的值， 然后取近似最大值
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 创建占位符 10列向量， 用于存储 实际分布正确值 ,然后对比预测的值和
y_ = tf.placeholder(tf.float32, [None, 10])

# 计算交叉熵 ,
# 现在了解为 比较两个结果的差异性 ， 即 比较  y 和  y_  的差异性
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 使用 gradient descent algorithm 的梯度下降算法,  --------------------------------对梯度下降算法的概念和逻辑都不太了解，  之后需要查询详细资料
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 调用函数 InteractiveSession   构建会话,  和 session类似 。  区别为 可以 运行在没有指定会话对象的情况下运行变量
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 进行 1000 此训练 ,
for _ in range(1000):
  # 每次取 100个数据进行训练
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # 执行梯度下降函数 , 训练模型
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 使用  argmax 函数预测正确性
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# 将布尔值转成浮点数， 取平均值   [True, False, True, True] --> [1,0,1,1] --> 0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 打印正确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))