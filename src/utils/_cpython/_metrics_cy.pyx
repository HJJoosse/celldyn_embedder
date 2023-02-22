import numpy as np
cimport numpy as np

np.import_array()

def trustworthiness(np.ndarray[np.int64_t, ndim=2] Q, Py_ssize_t K):
    """ The trustworthiness measure complements continuity and is a measure of
    the number of hard intrusions.

    Args:
        Q: the co-ranking matrix to calculate continuity from
        k (int): the number of neighbours to use.

    Returns:
        The trustworthiness metric for the given K
    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = Q.shape[0]
    cdef double summation = 0.0

    cdef double norm_weight = _tc_normalisation_weight(K, n+1)
    cdef double w = 2.0 / norm_weight
    
    for k in range(K, n):
        for l in range(K):
            summation += w * (k+1 - K) * Q[k, l]

    return 1.0 - summation

def continuity(np.ndarray[np.int64_t, ndim=2] Q, Py_ssize_t K):
    """ The continutiy measure complements trustworthiness and is a measure of
    the number of hard extrusions.

    Args:
        Q: the co-ranking matrix to calculate continuity from
        k (int): the number of neighbours to use.

    Returns:
        The continuity metric for the given K
    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = Q.shape[0]
    cdef double summation = 0.0

    cdef double norm_weight = _tc_normalisation_weight(K, n+1)
    cdef double w = 2.0 / norm_weight
    
    for k in range(K):
        for l in range(K, n):
            summation += w * (l+1 - K) * Q[k, l]

    return 1.0 - summation

def _tc_normalisation_weight(K, n):
    """ Compute the normalisation weight for the trustworthiness and continuity
    measures.

    Args:
        K (int): size of the neighbourhood
        n (int): total size of the matrix

    Returns:
        Normalisation weight for trustworthiness and continuity metrics
    """
    if K < (n/2):
        return n*K*(2*n - 3*K - 1)
    elif K >= (n/2):
        return n*(n - K)*(n - K)


def LCMC(np.ndarray[np.int64_t, ndim=2] Q, Py_ssize_t K):
    """ The local continuity meta-criteria measures the number of mild
    intrusions and extrusions. This can be thought of as a measure of the
    number of true postives.

    Args:
        Q: the co-ranking matrix to calculate continuity from
        k (int): the number of neighbours to use.

    Returns:
        The LCMC metric for the given K
    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = Q.shape[0]
    cdef double summation = 0.0

    for k in range(K):
        for l in range(K):
            summation += Q[k, l]
    
    return (K / (1. - n)) + (1. / (n*K)) * summation

def nMRRE(np.ndarray[np.int64_t, ndim=2] Q, Py_ssize_t K):
    """ The local continuity meta-criteria measures the number of mild
    intrusions and extrusions. This can be thought of as a measure of the
    number of true postives.

    Args:
        Q: the co-ranking matrix to calculate continuity from
        k (int): the number of neighbours to use.

    Returns:
        The nMRRE metric for the given K
    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = Q.shape[0]
    cdef double summation = 0.0

    for k in range(n-1):
        for l in range(K):
            summation += np.abs(k-l)/l*Q[k, l]

    for k in range(K):
        Hk += np.abs(n-2*k+1)/k
    Hk *= n

    return 1/Hk * summation

def vMRRE(np.ndarray[np.int64_t, ndim=2] Q, Py_ssize_t K):
    """ The local continuity meta-criteria measures the number of mild
    intrusions and extrusions. This can be thought of as a measure of the
    number of true postives.

    Args:
        Q: the co-ranking matrix to calculate continuity from
        k (int): the number of neighbours to use.

    Returns:
        The vMRRE metric for the given K
    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = Q.shape[0]
    cdef double summation = 0.0

    for k in range(K):
        for l in range(n-1):
            summation += np.abs(k-l)/k*Q[k, l]

    for k in range(K):
        Hk += np.abs(n-2*k+1)/k
    Hk *= n

    return 1/Hk * summation

def Qnx(np.ndarray[np.int64_t, ndim=2] Q, Py_ssize_t K, scaled=False):
    """ 
    The Qnx according to J.A. Lee, M. Verleysen: https://doi.org/10.1016/j.neucom.2008.12.017
    scaling following Lee et al: https://doi.org/10.1016/j.neucom.2014.12.095

    Args:
        Q: the co-ranking matrix to calculate continuity from
        k (int): the number of neighbours to use.
        scaled (bool): whether to scale the metric

    Returns:
        The QnxLV metric for the given K
    """
    cdef Py_ssize_t i, j

    n = Q.shape[0]
    if scaled:
        return ((n-1)*(LCMC(Q,K)+K/(n-1))-K)/(n-1-K)
    else:
        return LCMC(Q,K)+K/(n-1)




