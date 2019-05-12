import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

PIC = Image.open("ai.jpg", 'r')

piv_value = np.array(PIC)

print(piv_value.shape)


np.linalg.svd(piv_value[:, :, 0])

u_r, sigma_R, v_r = np.linalg.svd(piv_value[:, :, 0])
u_g, sigma_G, v_g = np.linalg.svd(piv_value[:, :, 1])
u_b, sigma_B, v_b = np.linalg.svd(piv_value[:, :, 2])
sigma_r=np.diag(sigma_R)
sigma_g=np.diag(sigma_G)
sigma_b=np.diag(sigma_B)

print(u_r.shape)

print(sigma_R.shape)

print(v_r.shape)


K=10
R=np.rint(np.dot(np.dot(u_r[:,0:K],sigma_r[0:K,0:K]),v_r[0:K,:])).astype('uint8')
R[R<0]=0;R[R>255]=255
G=np.rint(np.dot(np.dot(u_g[:,0:K],sigma_g[0:K,0:K]),v_g[0:K,:])).astype('uint8')
G[G<0]=0;G[G>255]=255
B=np.rint(np.dot(np.dot(u_b[:,0:K],sigma_b[0:K,0:K]),v_b[0:K,:])).astype('uint8')
B[B<0]=0;B[B>255]=255
I= np.stack((R, G, B), axis=2)

plt.imshow(I)
plt.show()