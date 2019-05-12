# 网络为 f(x;W,c,w,b) = w^Tmax{0,W^Tx + c} + b
import tensorflow as tf
sess = tf.Session()
x = tf.constant([[0,0],[0,1],[1,0],[1,1]]);
# 权重矩阵
W = tf.constant([[1,1],[1,1]])
# 偏置量 c
c = tf.constant([0,-1] ,shape=[1,2])
# 权重向量
w = tf.constant([1,-2],shape=[2,1])
# 与权重矩阵相乘
matmul_x = tf.matmul(x , W)
# 结果与0取最大值
# max_x = tf.maximum(matmul_x , 0)
#加上偏置量c
matmul_x_c = matmul_x+c
# 通过max 函数取最大值
max = tf.maximum(matmul_x_c , 0)
# 与权重向量相乘
result = tf.matmul(max , w)
# print(sess.run(matmul_x))
print(sess.run(result))






