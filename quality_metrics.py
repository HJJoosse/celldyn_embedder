# trustworthiness --> sklearn
# distance correlation correlation --> zelf chefffen
# knn-overlap - distance curve and integral
# poincarre


### knn_overlap -> find knn in true space random sample. Then embedding do the same for indices -> jaccard score

import scipy as sc
from sklearn.manifold import trustworthiness
from sklearn.metrics import jaccard_score
import dcor

class CDEmbeddingPerformance:

    def __init__(self,X_org:np.array,X_emb:np.array,metric:function):
        self.X_org = X_org
        self.X_emb = X_emb
        self.metric = metric

    def _return_trustworthiness(self):
        return trustworthiness(self.X_org,self.X_emb)

    def _create_knn_search(self):
        pass
    
    def _return_knn_overlap(self):
        pass

    def _distance_correlation(self):
        dist_before = sc.spatial.distance.pdist(self.X_org,metric = self.metric)
        dist_after = sc.spatial.distance.pdist(self.X_emb,metric = self.metric)

        return dcor.distance_correlation(dist_before, dist_after)
            
    def score(self):
        pass
