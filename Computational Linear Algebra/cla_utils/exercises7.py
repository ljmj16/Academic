import numpy as np
import cla_utils

def perm(p, i, j):
    """
    For p representing a permutation P, i.e. Px[i] = x[p[i]],
    replace with p representing the permutation P_{i,j}P, where
    P_{i,j} exchanges rows i and j.

    :param p: an m-dimensional numpy array of integers.
    """
    p[i],p[j] = p[j],p[i]
    return p


def LUP_inplace(A, ret_parity=False):
    """
    Compute the LUP factorisation of A with partial pivoting, using the
    in-place scheme so that the strictly lower triangular components
    of the array contain the strictly lower triangular components of
    L, and the upper triangular components of the array contain the
    upper triangular components of U.

    :param A: an mxm-dimensional numpy array

    :return p: an m-dimensional integer array describing the permutation \
    i.e. (Px)[i] = x[p[i]]
    """
    m = A.shape[0]
    #I introduce p   
    parity = 0             
    p = np.zeros(m,dtype=int)
    for l in range (m):
        p[l] = l
    #Now I start the loop    
    for k in range(m-1):
        #I choose j>=k to maximise |uik|
        j = np.argmax(np.abs(A[k:,k]))
        j = j+k
        if j !=k:
            parity +=1
        #Now I can swap the rows
        A[[k,j],:] = A[[j,k],:]
        #I use the perm function on p
        p = perm(p,j,k)
        #Now I use what I did in the previous exercise sheet
        A[k+1:,k] = A[k+1:,k] / A[k,k]
        A[k+1:,k+1:] = A[k+1:,k+1:] - np.outer( A[k+1:,k],  A[k,k+1:])
    if ret_parity == True:
        return p, parity

    
    return p


def solve_LUP(A, b):
    """
    Solve Ax=b using LUP factorisation.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an m-dimensional numpy array
    """

    m = A.shape[0]
    #I use the previous function to compute the in place LUP factorisation                 
    p = LUP_inplace(A)
    #I can obtain U and L
    L = np.eye(m) + np.tril(A,k=-1) 
    U = np.triu(A)
    #Now I have to apply the permutation to b
    b = b.reshape(m,1)
    b = b[p]
    b = b.reshape(m)
    #Now I use the functions solve_L and solve_U
    z = cla_utils.solve_L(L,b)
    x = cla_utils.solve_U(U,z)
    return x


def det_LUP(A):
    """
    Find the determinant of A using LUP factorisation.

    :param A: an mxm-dimensional numpy array

    :return detA: floating point number, the determinant.
    """
    m = A.shape[0]
    #LUP factorisation and extration of L and U
    p, parity = LUP_inplace(A, True)
    I = np.eye(m)
    #Now I compute the determinant. det(A) = det(L)*det(U) = det(U), which can be computed as the product of the elements of the diagonal
    detA = 1
    for i in range(m):
        detA = detA*A[i,i]
    if parity%2!=0:
        detA = -detA    
    return detA    
