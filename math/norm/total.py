from numpy import *
v1 = mat([100, 50])
v2 = mat([1000, 500])
# v1 = mat([2, 10])
# v2 = mat([10, 50])
print(sqrt((v1-v2)*(v1-v2).T))
print(dot(v1, v2.T)/(linalg.norm(v1, 2) * linalg.norm(v2)))

v1 = mat([2, 10])
v2 = mat([10, 50])

print(sqrt((v1-v2)*(v1-v2).T))
print(dot(v1, v2.T)/(linalg.norm(v1, 2) * linalg.norm(v2)))

