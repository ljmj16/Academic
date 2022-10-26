import numpy as np
import cla_utils
from scipy.linalg import solve_triangular

#regression with classical gram schmidt QR factorisation

v=np.arange(0.,1.0001,1./51)
m = v.shape
y=np.zeros(m)
y[0] = 1
y[50] = 1
X = np.vander(v, 13, increasing = True)
B = cla_utils.householder_ls(X, y)

print (B)