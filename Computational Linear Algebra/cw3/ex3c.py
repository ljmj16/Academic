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
 
A = cw3.ex3b.creationofA(8)

def scriptc(A):
    '''Takes a symmetric matrix A and computes the instructions given in ex 3c'''
    m,_ = A.shape
    A = cla_utils.hessenberg(A)
    v = np.zeros(m,dtype=complex)
    for i in reversed(range(m)):
        A = cla_utils.pure_QR(A,1000,0,case=True, case2 = False, case3=False)
        #I store the Tkk
        v[i] = A[i,i]
        A = A[:i,:i]
    return A,v


def scriptcmodified(A):
    '''Takes a symmetric matrix A and computes the instructions given in ex 3c and concatenates the Tkk-1'''
    m,_ = A.shape
    A = cla_utils.hessenberg(A)
    v = np.zeros(m,dtype=complex)
    list = []
    for i in reversed(range(m)):
        A,u = cla_utils.pure_QR(A,1000,0,case=True, case2 = True)
        #I store the Tkk
        v[i] = A[i,i]
        A = A[:i,:i]
        #I store the Tkk-1
        if i!=0:
            list = [*list,*u]
    return A,v,list

B,v,list = scriptcmodified(A)

n = len(list)

print(list)
plt.plot([i for i in range (n)],list)
plt.yscale('log')
plt.xlabel('')
plt.ylabel('')
plt.show()

def auxQR():
    A=cw3.ex3b.creationofA(10)
    B = cla_utils.hessenberg(A)
    C = cla_utils.pure_QR(B,1000,0,case=True)
    eigenvalues2 = np.diagonal(C)

def auxmodQR():
    A=cw3.ex3b.creationofA(10)
    A = scriptc(A)

def timingconf():
    print(timeit.Timer(auxQR).timeit(number=1))
    print(timeit.Timer(auxmodQR).timeit(number=1))

timingconf()

#check how precise the eigevalue are
#print(np.linalg.det(A-v[3]*np.ones((5,5))))

# For random symemetric matrices
S = random.uniform(size=(15,15))
S = (S+S.T)/2
S,p,nist = scriptcmodified(S)