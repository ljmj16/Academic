import numpy as np


def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """
    #dimensions
    m,n=Q.shape

    #starting a vector u and r
    u=np.zeros(n,dtype='complex')
    r=np.zeros(m,dtype='complex')
    #write u as the vector with components the inner product between v and qi, and implement r
    for i in range(n):
        u[i]=np.inner(np.conj(Q[:,i]),v)
        r-= u[i]*Q[:,i]
    r=r+v
    return r, u



def solveQ(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """
    #write the inverse of Q using the fact that Q is unitary
    A=np.conj(Q.T)
    x=np.dot(A,b)

    return x

def time_solveQ():
    """
    Function that computes the times required to solve Qx=b using solveQ
    for an input matrix of size 100 and the numpy builtin algorithm. 
    """
    
    #create random matrix Q with size 100 and random vector b 
    m = 100 
    A = np.random.rand(m,m)
    b = np.random.rand(m)
    #QR factorisation with the buildin algorithm
    Q, R = np.linalg.qr(A)
    #Time to solve Q with the custom algorithm
    t_0 = time.time()   
    x_1 = solveQ(Q,b)
    t_1 = time.time()
    print('time needed to solve Q with solveQ algorithm: ', t_1-t_0)
    #Time to solve Q with the builtin algorithm
    t = time.time()
    x_2 = np.linalg.solve(Q,b)
    t_1 = time.time()
    print('time needed to solve Q with np bultin algorithm: ', t_1-t_0)




def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """

    #we use the fact that Q is an orthogonal matrix
    C=np.conj(Q.T)
    P=Q.dot(C)

    return P
    

def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an mxl-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U, for appropriate l.
    """

    C, R = np.linalg.qr(V, mode='complete')

    m = V.shape[0]
    r = np.linalg.matrix_rank(V)

    Q = C[:,r:m] 

    return Q


def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm, transforming A to Q in place and returning R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """

    m, n = A.shape
    R = np.zeros((n,n), dtype=complex)
    #It works in place but I still need to introduce an array to store R as we have done above
    for j in range(n):
        v = A[:,j]
        for i in range(0,j):
            R[i][j] = np.dot( np.conj(A[:,i]), A[:,j])
            v = v - R[i][j]*A[:,i]
        R[j][j] = np.linalg.norm(v)
        A[:,j] = v / R[j][j]   


    return R

def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, transforming A to Q in place and returning
    R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """
    m,n = A.shape
    R = np.zeros((n,n), dtype=complex)
    for i in range (n):
        R[i,i] = np.linalg.norm(A[:,i])
        A[:,i] = A[:,i]/R[i,i]
        R[i,i+1:] = np.dot(A[:,i+1:].T, A[:,i].conjugate())
        A[:,i+1:] = A[:,i+1:] - np.outer(A[:,i], R[i, i+1:])
    
    return R


def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is normalised and orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """
    n=A.shape[1]
    R = np.eye(n, dtype=A.dtype)
    for i in range (k, n):
      R[k,i] = np.dot(np.conj(A[:,k].T) , A[:,i])

    return R

def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    A = 1.0*A
    R = np.eye(n, dtype=A.dtype)
    for i in range(n):
        Rk = GS_modified_get_R(A, i)
        A[:,:] = np.dot(A, Rk)
        R[:,:] = np.dot(R, Rk)
    R = np.linalg.inv(R)
    return A, R
