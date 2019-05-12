# import tensorflow as tf
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))


import tensorflow as tf
from tensorflow import keras

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

r = math.log(2, 2)*1/4

r1 = math.log(2, 2)*1/4
print(1-(r+r1)*1/2)


print(tf.__version__)