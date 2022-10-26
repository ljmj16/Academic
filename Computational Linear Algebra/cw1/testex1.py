import numpy as np
import pytest
import ex1
import cla_utils
import copy

# I am using this test to check the compression of the matrix C
def test_1_1():
    C = np.loadtxt('C:\\Users\\Lucjano & Klara\\Desktop\\C.dat', delimiter=',')
    # I am copying C since I need it to confront with the compressed C and the function I am using changes in place
    D = copy.deepcopy(C)
    C_com = ex1.compression(C)
    assert(np.linalg.norm(D-C_com) < 1.0e-8)
    
