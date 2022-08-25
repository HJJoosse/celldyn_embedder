from numba import njit
import numpy as np
from numpy import linalg as la


@njit
def poincarre_dist(x,y):
    return np.arccosh(\
    1 + 2*(\
        la.norm(x-y, ord=1)**2/((1-la.norm(x, ord=1)**2)*(1-la.norm(y, ord=1)**2))
        )
    )