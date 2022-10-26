import numpy as np
from numpy.core.fromnumeric import transpose

def Q1AQ1s(A):
    """
    For a matrix A, find the unitary matrix Q1 such that the first
    column of Q1*A has zeros below the diagonal. Then return A1 = Q1*A*Q1^*.

    :param A: an mxm numpy array

    :return A1: an mxm numpy array
    """
    #A is a mxm matrix
    m = A.shape[0]
    #I use a loop 
    for k in range(0,m):
        x = A[k:m,k].copy()
        # Implementation of e_1
        e_1 = np.zeros((m-k),dtype=float)
        e_1[0] = 1
        # sign of x_1
        l = np.sign(x[0]) if x[0] else 1
        # v and A
        v = l*np.linalg.norm(x)*e_1 + x
        v = v / np.linalg.norm(v)
        #performing multiplication from the left and the right to compute the matrix A1
        A[k:m,k:m] -= 2*np.outer( v, np.dot( v.T, A[k:m,k:m]))
        v = v.reshape(m-k,1)
        A[:,k:m] -= 2*np.dot(np.dot(A[:,k:m],v), np.conjugate(v.transpose()))
    return A


def hessenberg(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.

    :param A: an mxm numpy array
    """
    #A is a mxm matrix
    m = A.shape[0]
    #I use a loop 
    for k in range(0,m-2):
        x = A[k+1:m,k].copy()
        # Implementation of e_1
        e_1 = np.zeros((m-k-1))
        e_1[0] = 1
        # sign of x_1
        l = x[0]/np.linalg.norm(x[0]) if x[0] else 1
        # v and A
        v = l*np.linalg.norm(x)*e_1 + x
        v = v / np.linalg.norm(v)
        #performing the algorithm
        v = v.reshape(m-k-1,1)
        A[k+1:m,k:m] -= 2*np.outer( v, np.dot( np.conjugate(v.transpose()), A[k+1:m,k:m]))
        A[:m,k+1:m] -= 2*np.outer(np.dot(A[:m,k+1:m],v), np.conjugate(v.transpose()))
    return A


def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array
    
    :return Q: an mxm numpy array
    """

    #A is a mxm matrix
    m = A.shape[0]
    Qstar = np.eye(m)
    #I use a loop 
    for k in range(0,m-2):
        x = A[k+1:m,k].copy()
        # Implementation of e_1
        e_1 = np.zeros((m-k-1),dtype=float)
        e_1[0] = 1
        # sign of x_1
        l = np.sign(x[0]) if x[0] else 1
        # v and A
        v = l*np.linalg.norm(x)*e_1 + x
        v = v / np.linalg.norm(v)
        #performing the algorithm
        v = v.reshape(m-k-1,1)
        A[k+1:m,k:m] -= 2*np.dot( v, np.dot( np.conjugate(v.transpose()), A[k+1:m,k:m]))
        A[:m,k+1:m] -= 2*np.dot(np.dot(A[:m,k+1:m],v), np.conjugate(v.transpose()))
        #writing Q*
        Qstar[k+1:m,:] -= 2*np.dot(np.dot(v, np.conjugate(v.transpose())), Qstar[k+1:,:])
        #writing Q
        Q = np.conjugate(Qstar.transpose())
    return Q



def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvalues and eigenvectors.

    :param H: an mxm numpy array

    :return ee: an m dimensional numpy array containing the eigenvalues of H
    :return V: an mxm numpy array whose columns are the eigenvectors of H
    """
    m, n = H.shape
    assert(m==n)
    assert(np.linalg.norm(H[np.tril_indices(m, -2)]) < 1.0e-6)
    _, V = np.linalg.eig(H)
    return V


def ev(A):
    """
    Given a matrix A, return the eigenvectors of A. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).

    :param A: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """
    Q = hessenbergQ(A)
    V = hessenberg_ev(A)
    V = np.dot(Q,V)
    return V
