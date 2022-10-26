import numpy as np
import pytest
import scipy
import cla_utils
import cw3
from cw3.ex3b import creationofA
from cw3.ex3c import scriptc
from cw3.ex3d import scriptdmodified3d


@pytest.mark.parametrize('m', [5, 10, 7])
def test_ex3c(m):
    A = creationofA(m)
    u = np.linalg.eig(A)[0]
    A,v,l = scriptdmodified3d(A)
    assert(np.linalg.norm(u-v)<1.0e-7)