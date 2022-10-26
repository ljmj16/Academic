import numpy as np
from numpy.lib.function_base import copy


def get_Lk(m, lvec):
    """Compute the lower triangular row operation mxm matrix L_k 
    which has ones on the diagonal, and below diagonal entries
    in column k given by lvec (k is inferred from the size of lvec).

    :param m: integer giving the dimensions of L.
    :param lvec: a m-k dimensional numpy array.
    :return Lk: an mxm dimensional numpy array.

    """
    L_k = np.eye(m)
    n = len(lvec)
    #determine k
    k = m-n
    #form L_k
    L_k[k:,k-1] = -lvec 
    return L_k


def LU_inplace_first(A):
    """Compute the LU factorisation of A, using the in-place scheme so
    that the strictly lower triangular components of the array contain
    the strictly lower triangular components of L, and the upper
    triangular components of the array contain the upper triangular
    components of U.

    :param A: an mxm-dimensional numpy array

    """
    m = A.shape[0]
    #now I can start the loop
    for k in range(m-1):
        #inner loop
        for j in range(k+1,m):
            ljk = A[j,k]/ A[k,k]
            A[j,k:m] = A[j,k:m]-ljk*A[k,k:m]
            A[j,k] = ljk 
    return A       

def LU_inplace(A):
    """
    previous function written in a single loop using outer product
    """
    m = A.shape[0]
    #now I can start the loop
    for k in range(m-1):
        A[k+1:,k] = A[k+1:,k] / A[k,k]
        A[k+1:m,k+1:m] = A[k+1:m,k+1:m] - np.outer( A[k+1:m,k],  A[k,k+1:m])
    return A

def solve_L(L, b):
    """
    Solve systems Lx_i=b_i for x_i with L lower triangular, i=1,2,...,k

    :param L: an mxm-dimensional numpy array, assumed lower triangular
    :param b: an mxk-dimensional numpy array, with ith column containing 
       b_i
    :return x: an mxk-dimensional numpy array, with ith column containing 
       the solution x_i

    """
    m,_ = L.shape
    

    #I initialize a matrix of zeros
    x = np.empty(b.shape) 
    #first step
    x[0] = b[0] / L[0, 0] 
    #loop forward
    for i in range(1, m): 
        alpha = 1 / L[i,i]
        x[i] = alpha*(b[i] - np.dot(L[i,0:i],x[0:i]))   

    return x


def solve_U(U, b):
    """
    Solve systems Ux_i=b_i for x_i with U upper triangular, i=1,2,...,k

    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing 
       b_i
    :return x: an mxk-dimensional numpy array, with ith column containing 
       the solution x_i

    """
    m,_ = U.shape

    #I initialize a vector of zeros
    x = np.empty(b.shape) 
    #first step
    x[m-1] = b[m-1] / U[m-1,m-1] 
     #loop backwards
    for i in range(m-2,-1,-1):
        alpha = 1 / U[i,i]
        x[i] = alpha*(b[i] - np.dot(U[i,i+1:],x[i+1:]))    
                
    return x


def inverse_LU(A):
    """
    Form the inverse of A via LU factorisation.

    :param A: an mxm-dimensional numpy array.

    :return Ainv: an mxm-dimensional numpy array.

    """
    #I have to obtain L and U from A first
    m = A.shape[0]
    A = LU_inplace(A)
    #It's easy to extract A
    U = np.triu(A) 
    I = np.eye(m) 
    #I sum the identity to the 'stricly' lower part of A
    L = I + np.tril(A, k=-1) 
    #Now I can compute the algorithm
    Y = solve_L(L,I) # I use this function to solve LY=I
    X = solve_U(U,Y) #I use this function to solve UX=Y
    #Now for the LU factorisation I have LY=LUX=AX=I. So I have solved AX=I which gives me the Ainv (which I call x)
    return X
