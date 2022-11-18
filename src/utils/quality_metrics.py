# trustworthiness --> sklearn
# distance correlation correlation --> zelf chefffen
# knn-overlap - distance curve and integral
# poincarre


### knn_overlap -> find knn in true space random sample. Then embedding do the same for indices -> jaccard score

import scipy as sc
from scipy.spatial.distance import jaccard, hamming
import numpy as np
from sklearn.manifold import trustworthiness
import dcor
import faiss

class CDEmbeddingPerformance:

    def __init__(self,X_org:np.array,X_emb:np.array,metric,n_neighbours:int=10):
        self.X_org = X_org
        self.X_emb = X_emb
        self.metric = metric
        self.n_neighbours = n_neighbours

    def _return_trustworthiness(self):
        trustworth_score = []
        for _ in range(10):
            sample = np.random.choice(np.arange(len(self.X_org)),size = 1000)
            X_org = self.X_org[sample,:]
            X_emb = self.X_emb[sample,:]
            trustworth_score.append(trustworthiness(X_org,X_emb,n_neighbors=self.n_neighbours,metric = self.metric))
        return trustworth_score

    @staticmethod
    def _create_knn_search(X,k):
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X.astype(np.float32))
        D,I = index.search(X.astype(np.float32), k)
        return D,I
    
    def _return_knn_overlap(self):
        D,I = self._create_knn_search(self.X_org,self.n_neighbours)
        D_emb,I_emb = self._create_knn_search(self.X_emb,self.n_neighbours)
        jaccard_ds, hamming_ds = np.zeros(I.shape[0]), np.zeros(I.shape[0])
        for i in range(I.shape[0]):
            jaccard_ds[i] = jaccard(I[i,:],I_emb[i,:])
            hamming_ds[i] = hamming(I[i,:],I_emb[i,:])
        return jaccard_ds, hamming_ds

    def _distance_correlation(self):
        dist_correlation = []
        for _ in range(10):
            sample = np.random.choice(np.arange(len(self.X_org)),size = 1000)
            X_org = self.X_org[sample,:]
            X_emb = self.X_emb[sample,:]
            dist_before = sc.spatial.distance.pdist(X_org,metric = self.metric)
            dist_after = sc.spatial.distance.pdist(X_emb,metric = self.metric)
            dist_correlation.append(dcor.distance_correlation(dist_before, dist_after))
        return dist_correlation
            
    def score(self):
        return self._return_knn_overlap(), self._distance_correlation(), self._return_trustworthiness()
