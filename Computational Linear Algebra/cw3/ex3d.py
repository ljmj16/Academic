import numpy as np
from numpy.core.fromnumeric import shape
import scipy
import cla_utils
from cla_utils.exercises9 import pure_QR
import cw3.ex3b
import matplotlib.pyplot as plt
import timeit
from numpy import random 


#matrix A
 
A = cw3.ex3b.creationofA(15)
C = random.uniform(size=(15,15))
C = (C+C.T)/2

def scriptdmodified3d(A):
    '''EX 3D version - Takes a symmetric matrix A and computes the instructions given in ex 3c and concatenates the Tkk-1'''
    m,_ = A.shape
    A = cla_utils.hessenberg(A)
    v = np.zeros(m,dtype=complex)
    list = []
    for i in reversed(range(m)):
        A,u = cla_utils.pure_QR(A,10000,0,case=True, case2 = True, case3 = True)
        #I store the Tkk
        v[i] = A[i,i]
        A = A[:i,:i]
        #I store the Tkk-1
        if i!=0:
            list = [*list,*u]
    return A,v,list

B,v,list = scriptdmodified3d(C)

n = len(list)

print(list)
plt.plot([i for i in range (n)],list)
plt.yscale('log')
plt.xlabel('')
plt.ylabel('')
plt.show()
