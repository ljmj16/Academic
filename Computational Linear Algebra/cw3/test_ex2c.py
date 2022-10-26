import numpy as np
import pytest
import cla_utils
import cw3
from cw3.ex2c import eigenvaluescalculator
from cw3.ex2 import matrixconstruction
from cw3.ex2d import matrixconstructionB

@pytest.mark.parametrize('m', [4, 10, 6])
def test_eigenchecker(m):
    A = matrixconstruction(m)
    A = cla_utils.pure_QR(A, 3000, 1.0e-7)
    #Now that I have applied QR I can find the eigenvalues with my algorithm
    v = eigenvaluescalculator(A)
    L = 0
    for i in range(2*m):
        L = np.linalg.det(A-v[i]*np.eye(2*m))
    #we check that the eigenvalues are correct
        assert(np.linalg.norm(L)<1.0e-5)

