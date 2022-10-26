import numpy as np
import cla_utils
from scipy.linalg import solve_triangular
import copy


v=np.arange(0.,1.0001,1./51)

def Regression_numpy(v):
   #implements the regression using numpy function for qr factorisation
   m = v.shape
   #y is an array with the values of f_i
   y=np.zeros(m)
   y[0] = 1
   y[50] = 1
   #We implement the Vandermonde matrix
   X = np.vander(v, 13, increasing = True)
   #We use the numpy qr builtin factorisation
   Q, R = np.linalg.qr(X)
   #we proceed as explained in the report to find B
   y_alt = np.dot(Q.T, y)
   B = solve_triangular(R, y_alt)
   return B



def Regression_GSclassical(v):
#implements the regression using GS classical for the qr factorisation
   m = v.shape
   y=np.zeros(m)
   y[0] = 1
   y[50] = 1
   X = np.vander(v, 13, increasing = True)
   R = cla_utils.GS_classical(X)
   y_alt = np.dot(X.T, y)
   B = solve_triangular(R, y_alt)
   return B


def Regression_GSmodified(v):
#implements regression using GS modified
   m = v.shape
   y=np.zeros(m)
   y[0] = 1
   y[50] = 1
   X = np.vander(v, 13, increasing = True)
   Q=copy.deepcopy(X)
   R = cla_utils.GS_modified(Q)
   y_alt = Q.transpose().dot(y)
   B = solve_triangular(R, y_alt)
   return B


def Regression_householder(v):
#implements regression using householder
   m = v.shape
   y=np.zeros(m)
   y[0] = 1
   y[50] = 1
   X = np.vander(v, 13, increasing = True)
   B = cla_utils.householder_ls(X, y)
   return B