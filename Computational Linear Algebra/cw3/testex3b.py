import numpy as np
import pytest
import scipy
import cla_utils
import cw3
from cw3.ex3b import creationofA


@pytest.mark.parametrize('m', [5, 15, 25])
def test_ex3btriu(m):
    A = creationofA(m)
    l = np.linalg.eig(A)[0]
    B = scipy.linalg.hessenberg(A)
    C = cla_utils.pure_QR(B,1000,0,case=True)
    eigenvalues2 = np.diagonal(C)
    assert(np.linalg.norm(eigenvalues2-l)<0.5)



