import numpy as np
import cla_utils
import ex2
from cw3.ex2 import matrixconstruction

def eigenvaluescalculator(A):
    '''calculates the eigenvalues from the output of the pureQR applied to the matrix A'''
    m,_ = A.shape
    #we define a vector where to store our eigenvalues
    v = np.zeros(m, dtype= complex)
    k = np.arange(m-1, step = 2)
    l = np.arange(1, m, step = 2)
    i = 0
    #for loop that takes the elemnts and obtains the eigenvalues
    for elem in A[k,l]:
        v[i] = 1j*elem
        v[i+1] = -1j*elem 
        i+= 2
    return v

A = matrixconstruction(2)
A = cla_utils.pure_QR(A,100,1.0e-4)
v = eigenvaluescalculator(A)
print(np.linalg.eig(A))