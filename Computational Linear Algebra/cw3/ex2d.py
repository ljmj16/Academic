import numpy as np
import cla_utils
import matplotlib.pyplot
import pytest



def matrixconstructionB(n):
    '''Construct a matrix B 2nx2n, like the one requested in the exercise 2 of the coursework'''
    B = np.zeros((2*n,2*n))
    B += np.diag(2*np.ones(2*n-1),k=1)
    B += np.diag(-np.ones(2*n-1),k=-1)
    return B


def eigenvaluescalculatorB(B):
    '''calculates the eigenvalues from the output of the pureQR applied to the matrix B'''
    m,_ = B.shape
    #we define a vector where to store our eigenvalues
    v = np.zeros(m, dtype= complex)
    k = np.arange(m-1, step = 2)
    l = np.arange(1,m, step = 2)
    i = 0
    #for loop that takes the elemnts and obtains the eigenvalues
    for elem in B[k,l]*B[l,k]:
        v[i] = np.sqrt(elem)
        v[i+1] = -np.sqrt(elem) 
        i+= 2
    return v

@pytest.mark.parametrize('m', [4, 10, 6])
def test_eigenchecker(m):
    B = matrixconstructionB(m)
    B = cla_utils.pure_QR(B, 3000, 1.0e-7)
    #Now that I have applied QR I can find the eigenvalues with my algorithm
    v = eigenvaluescalculatorB(B)
    L = 0
    for i in range(2*m):
        L = np.linalg.det(B-v[i]*np.eye(2*m))
    #we check that the eigenvalues are correct
        assert(np.linalg.norm(L)<1.0e-5)




B = matrixconstructionB(10)
#print(np.linalg.eig(A))
B = cla_utils.pure_QR(B,100,1.0e-6)
# to plot :
# matplotlib.pyplot.matshow(B.real)
# matplotlib.pyplot.show()
# print(B)
