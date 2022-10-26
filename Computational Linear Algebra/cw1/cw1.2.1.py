import numpy as np
import cla_utils
from scipy.linalg import solve_triangular
import copy

v=np.arange(0.,1.0001,1./51)
m = v.shape
y=np.zeros(m)
y[0] = 1
y[50] = 1
X = np.vander(v, 13, increasing = True)
Q, R = np.linalg.qr(X)
y_alt = Q.transpose().dot(y)
B = solve_triangular(R, y_alt)

print (B)