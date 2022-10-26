import numpy as np
import timeit
import numpy.random as random

# pre-construct a matrix in the namespace to use in tests
random.seed(1651)
A0 = random.randn(500, 500)
x0 = random.randn(500)



def basic_matvec(A, x): 
    """
    Elementary matrix-vector multiplication.

    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    returns an m-dimensional numpy array which is the product of A with x

    This should be implemented using a double loop over the entries of A

    :return b: m-dimensional numpy array
    """
    m,n=A.shape

    b=np.zeros(m)

    for i in range(m):
        for j in range(n):
           b[i]+=A[i,j]*x[j]
    return b
 





def column_matvec(A, x):
    """
    Matrix-vector multiplication using the representation of the product
    Ax as linear combinations of the columns of A, using the entries in 
    x as coefficients.


    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    :return b: an m-dimensional numpy array which is the product of A with x

    This should be implemented using a single loop over the entries of x
   """
    m,n=A.shape
    b=np.zeros(m)
    for i in range(n):
        b+=A[:,i]*x[i]
    return b



def timeable_basic_matvec():
    """
    Doing a matvec example with the basic_matvec that we can
    pass to timeit.
    """

    b = basic_matvec(A0, x0) # noqa


def timeable_column_matvec():
    """
    Doing a matvec example with the column_matvec that we can
    pass to timeit.
    """

    b = column_matvec(A0, x0) # noqa


def timeable_numpy_matvec():
    """
    Doing a matvec example with the builtin numpy matvec so that
    we can pass to timeit.
    """

    b = A0.dot(x0) # noqa


def time_matvecs():
    """
    Get timings for matvecs.
    """
    print("Timing for the basic_matvec")
    print(timeit.Timer(timeable_basic_matvec).timeit(number=1))
    
    print("Timing for the column_matvec")
    print(timeit.Timer(timeable_column_matvec).timeit(number=1))
   
    print("Timing for the numpy matvec")
    print(timeit.Timer(timeable_numpy_matvec).timeit(number=1))


def rank2(u1, u2, v1, v2):
    """
    Return the rank2 matrix A = u1*v2^* + u2*v2^*.

    :param u1: m-dimensional numpy array
    :param u2: m-dimensional numpy array
    :param v1: n-dimensional numpy array
    :param v2: n-dimensional numpy array
    """

    B= np.array(np.transpose([u1,u2]),dtype=complex)
    C=np.array([np.conj(v1),np.conj(v2)],dtype=complex)
    A=B.dot(C)
    return A


def rank1pert_inv(u, v):
    """
    Return the inverse of the matrix A = I + uv^*, where I
    is the mxm dimensional identity matrix, with

    :param u: m-dimensional numpy array
    :param v: m-dimensional numpy array
    """
    m=len(u)
    I=np.eye(m)
    alfa = 1/(1 + np.inner(u,np.conj(v)))
    B=np.outer(u,np.conj(v))
    Ainv=I-alfa*B
    return Ainv

def time_for_inv():

    u = np.random.rand(400) + 1j*np.random.rand(400)
    v = np.random.rand(400) + 1j*np.random.rand(400)
    A = np.eye(400) + np.outer(u, np.conj(v))

    t0 = time.time()
    rank1pert_inv(u,v)
    t1 = time.time()
    print('time needed for my inversion: ', t1-t0)
    t0 = time.time()
    np.linalg.inv(A)
    t1 = time.time()
    print('time needed by the numpy builtin inversion algorithm: ', t1-t0)    
    


def ABiC(Ahat, xr, xi):
    """Return the real and imaginary parts of z = A*x, where A = B + iC
    with

    :param Ahat: an mxm-dimensional numpy array with Ahat[i,j] = B[i,j] \
    for i>=j and Ahat[i,j] = C[i,j] for i<j.

    :return zr: m-dimensional numpy arrays containing the real part of z.
    :return zi: m-dimensional numpy arrays containing the imaginary part of z.
    """
    #definitions
    m=len(xi)
    zr=np.zeros(m)
    zi=np.zeros(m)
    B=np.zeros((m,m))
    C=np.zeros((m,m))
    #loop
    for i in range(m):
      
      B[i:m,i]=Ahat[i:m,i]
      B[i,i:m]=Ahat[i:m,i]
      
      C[i:m,i]=-Ahat[i,i:m]
      C[i,i:m]=Ahat[i,i:m]
      C[i,i]=0

      zi+=B[:,i]*xi[i]+C[:,i]*xr[i]
      zr+=B[:,i]*xr[i]-C[:,i]*xi[i]

    return zr, zi
