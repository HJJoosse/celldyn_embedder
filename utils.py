from numba import njit, float32
import numpy as np
from numpy import linalg as la


@njit(float32(float32[:],float32[:]), fastmath=True)
def poincarre_dist(x,y):
    return np.arccosh(\
    1 + 2*(\
        la.norm(x-y,ord = 2)**2/((1-la.norm(x,ord = 2)**2)*(1-la.norm(y,ord = 2)**2))
        )
    )

@njit(float32(float32[:],float32[:]), fastmath=True)
def fractional_distance(x, y, f=0.5):
    return np.power(np.abs(np.sum(np.power(x-y, f))), 1/f)