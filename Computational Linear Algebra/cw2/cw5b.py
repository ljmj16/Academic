import numpy as np
import cla_utils
from numpy import random
from scipy.sparse import diags
from cla_utils import LU_inplace
import timeit
import matplotlib.pyplot as plt

def bandedmatrixgenerator(n):
    '''Given n, this function generate a banded matrix (n-1)^2x(n-1)^2, with both lower and upper bandwidth equal to n-1. (which is a structur like the one of the matrix A of the ex 4b)'''
    l = (n-1)**2
    A = np.zeros((l,l))
    for i in range(n-1):
       A += np.diag(random.uniform(low=0,high=1,size=l-i),-i)
       A += np.diag(random.uniform(low=0,high=1,size=l-i),i)
    return A


def LUfactorisationbanded(A,p,q):
    '''Given a banded matrix with bandwidths p and q performs the LU factorisation for banded matrices'''
    m = A.shape[0]
    #now I can start the loop
    for k in range(m-1):
        A[k+1:min(k+p,m),k] = A[k+1:min(k+p,m),k] / A[k,k]
        A[k+1:min(k+p,m),k+1:min(k+q,m)] = A[k+1:min(k+p,m),k+1:min(k+q,m)] - np.outer( A[k+1:min(k+p,m),k],  A[k,k+1:min(k+q,m)])
    return A

def auxiliartiming():
    #generating the matrices
    A = bandedmatrixgenerator(n)
    #Runs LUfatorisationbanded for our particular matrices
    LUfactorisationbanded(A,n-1,n-1)

def timing():
    #first we define a list where we will store at each step our computaion time divided by the operation count we proposed in the exercise before (so n^4)
    list = [] 
    for n in range (10,670):
        #we generate bamded matrices
        n = timeit.Timer(auxiliartiming).timeit(number=7)/n**4
        list.append(n)
    #Now we plot it
    plt.plot([i+10 for i in range (10,670)],list)
    plt.xlabel('n')
    plt.ylabel('Time divided for n^4')
    plt.show()





#We check the error
n = 4
A = bandedmatrixgenerator(n)
A0 = LU_inplace(A)
A = LUfactorisationbanded(A,n-1,n-1)
err = np.linalg.norm(A-A0)
#we print the error
print (err)
#we plot the timings
timing()


