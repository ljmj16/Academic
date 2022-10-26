import pytest
import cla_utils
from numpy import random
import numpy as np

from cw3.ex4b import H_apply


@pytest.mark.parametrize('m', [(5), (21), (7), (10)])
def test_happly(m):
    random.seed(8578*m)
    v = random.randn(m**2)
    laplacian, sumtest = H_apply(v,m,5,2,case=True)
    assert(np.linalg.norm(sumtest)<1.0e-9) #the test is the one suggested in the notes
