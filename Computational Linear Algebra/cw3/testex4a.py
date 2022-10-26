import pytest
import cla_utils
from numpy import random
import numpy as np

from cw3.ex4a import fastserialiser, reformer





@pytest.mark.parametrize('m,n', [(5,2), (21,5), (80,7), (7,5)])
def test_flattenerandreformer(m,n):
    random.seed(8578*m+1*n)
    A = random.randn(m,n)
    v = fastserialiser(A)
    B = reformer(v,m,n)
    assert(np.linalg.norm(B-A)<1.0e-6)
