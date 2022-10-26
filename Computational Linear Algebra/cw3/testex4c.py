import pytest
import cla_utils
from numpy import random
import numpy as np
from cla_utils.exercises10 import GMRES
from cw3.ex4a import reformer

from cw3.ex4b import H_apply

@pytest.mark.parametrize('m', [5])
def test_GMRES(m):
    random.seed(8578*m)
    b = random.randn(m**2)
    x = GMRES(None,b,1000,0.000001,situation = H_apply)[0]
    assert(np.linalg.norm(H_apply(x,case=False)-b)<1.0e-6)

b = random.randn(5**2)
x = GMRES(None,b,1000,0.000001,situation = H_apply)
print(x)