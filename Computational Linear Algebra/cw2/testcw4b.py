import numpy as np
from numpy import random
import cla_utils
import pytest
import cw2
from cw2.cw4b import creation_matrix
from cw2.cw4b import bandedalgorithm


@pytest.mark.parametrize('m', [28, 90, 27, 15, 80, 78, 110, 3, 2, 6, 3])
def test_banded_algorithm(m):
    random.seed(1765)
    A = creation_matrix(m)
    #Copy of A
    A0 = 1.0*A
    #Applyig the algorithm
    A = bandedalgorithm(A)
    #extracting L and U
    U = np.triu(A) 
    L = np.eye(4*m+1) + np.tril(A, k=-1) 
    #Now I run the test
    assert (np.linalg.norm(A0-np.dot(L,U))< 1.0e-6)

