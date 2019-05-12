import numpy as np
import matplotlib.pylab as plt

def linear_regersssion(x, y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x**2)
    sumxy = sum(x*y)
    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])
    print(np.linalg.solve(A, b))
    return np.linalg.solve(A, b)


X = np.arange(0, 10, 0.1)
Z = [8 + 7 * x for x in X]
Y = [np.random.normal(z, 4) for z in Z]
plt.plot(X, Y, 'k.')
# plt.show()

a, b = linear_regersssion(X, Z)

# print(a,b)
# plt.plot(X, a*X + b, 'r')
plt.show()

# 注 ，  这里的G为   X , b 为 Y  返回
bv = np.ones(len(X))
G = np.mat(np.array([X, bv]).T)
G.shape
b = np.matrix(Z).T
b.shape
print((G.T*G).I*(G.T)*b)
