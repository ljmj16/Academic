import numpy as np
import pytest
from numpy import random
from cla_utils.exercises10 import GMRES

from cw3.ex4d import M_solve


@pytest.mark.parametrize('m',[2,4,7,11])
def test_ex4e(m):
    random.seed(8578*m)
    A = random.uniform((m,m))
    b = random.randn(m)
    precondit = M_solve
    x = GMRES(A,b,1000,1.0e-6, x0=None,preconditioning = precondit)[0]
    assert(np.linalg.norm(A@x-b)<1.0e-6)
