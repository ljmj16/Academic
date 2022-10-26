import numpy as np
import cla_utils
import scipy
 
def householderVR(A, kmax=None):
   m, n = A.shape

   if kmax is None:
        kmax = n
    #I use a loop 
   for k in range(0,kmax):
        x = A[k:m,k].copy()
        # Implementation of e_1
        e_1 = np.zeros((m-k),dtype=float)
        e_1[0] = 1
        # sign of x_1
        if x[0]  == 0:
            l = 1,
        else:
            l = np.sign(x[0])
        # v and A
        v = l*np.linalg.norm(x)*e_1 + x
        v = v / np.linalg.norm(v)
        A[k:m,k:n] = A[k:m,k:n] - 2*np.outer( v, np.dot( v.T, A[k:m,k:n]))


        #I 'adapt' v to make the first element of each vectors the same as the elements of the diagonal of the matrix
        v = (A[k,k]/v[0])*v
        #then I write the v vectors in A(in particulare in the lower triangular part of A, including the diagonal (which doesn't change at all in this computation))
        A[k:,k] = v
    # I copy A in R
   R = A[:,:]
   return R

def multiplicationQstarb(R_v,b):
    m,n=R_v.shape
    for i in range(n):
        v = R_v[i:,i]
        v = v / np.linalg.norm(v)
        b[i:m] = b[i:m]-2*np.dot( v, np.dot(v.T,b[i:m]))
    return b


def solvels(R_v,b):
   m,n = R_v.shape
   Q_starb = multiplicationQstarb(R_v,b)
   #now we take the first n elements 
   Q_hat_starb = Q_starb[:n]
   #we use a numpy builtin function to get R
   R0 = np.triu(R_v)
   R1 = R0[:n,:n]
   x = scipy.linalg.solve_triangular(R1,Q_hat_starb)
   return x
