from numba import njit, float32
import numpy as np
from numpy import linalg as la


@njit(float32(float32[:],float32[:]))
def poincarre_dist(x,y):
    return np.arccosh(\
    1 + 2*(\
        la.norm(x-y,ord = 2)**2/((1-la.norm(x,ord = 2)**2)*(1-la.norm(y,ord = 2)**2))
        )
    )