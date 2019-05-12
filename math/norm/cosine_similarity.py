# 余弦相似度
from numpy import *


v1 = mat([1, 2, 2, 1, 1, 1, 0])
v2 = mat([1, 2, 2, 1, 1, 2, 1])

# v1 = mat([1, 1, 2, 1, 1, 1, 0, 0, 0])
# v2 = mat([1, 1, 1, 0, 1, 1, 1, 1, 1])

print(dot(v1, v2.T)/(linalg.norm(v1, 2) * linalg.norm(v2)))


