import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

__author__ = "Joaquim Viegas"
"""
Python implementation of Dunn index by Joaquim Viegas

dunn_fast(points, labels):
    Depends on numpy for fast Dunn index calculation using pairwise distances
    calculated by sklearn. 

"""

def delta_fast(ck, cl, distances):
    
    """
    Find the min distance of any two clusters 
    Parameters
    -----------
    ck: index of first cluster being compared
    cl: labels of second cluster being compared
    distance: pairwise distance of all points

    """
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]

    return np.min(values)
    
def big_delta_fast(ci, distances):
    
    """
    Find the max distance between points in the cluster  
    Parameters
    -----------
    ci: index of the cluster being compared
    distance: pairwise distance of all points

    """
    values = distances[np.where(ci)][:, np.where(ci)]
    #values = values[np.nonzero(values)]
            
    return np.max(values)

def dunn_fast(points, labels):
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)
    
    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = euclidean_distances(points)
    ks = np.sort(np.unique(labels))
    
    deltas = np.ones([len(ks), len(ks)])*1000000
    big_deltas = np.zeros([len(ks), 1])
    
    l_range = list(range(0, len(ks)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            
            #min distances of any two clusters
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances) 
        
        #max distances of two points in a cluster
        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)
    
    di = np.min(deltas)/np.max(big_deltas)
    return di