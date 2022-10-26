import numpy as np
import cla_utils

C = np.loadtxt('C:\\Users\\Lucjano & Klara\\Desktop\\C.dat', delimiter=',')


def compression(C):
    #we compute the QR factorisation
    Q, R = cla_utils.householder_qr(C)
    #we use Q_com and R_com (see the report for explaination) to get C_com
    C_com= Q[:,:3].dot(R[:3,:])
    return(C_com)


