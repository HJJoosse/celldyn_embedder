# trustworthiness --> sklearn
# distance correlation correlation --> zelf chefffen
# knn-overlap - distance curve and integral
# poincarre


### knn_overlap -> find knn in true space random sample. Then embedding do the same for indices -> jaccard score
import scipy as sc
from scipy.spatial.distance import jaccard, hamming
import numpy as np
from numpy.random import default_rng
from sklearn.manifold import trustworthiness
import dcor
import faiss
import time
from tabulate import tabulate
from collections import defaultdict


class CDEmbeddingPerformance:

    def __init__(self,metric='euclidean',n_neighbours:int=10, knn_dist:str='jaccard', knn_return_median:bool = True, num_triplets:int=5):
        self.metric = metric
        self.n_neighbours = n_neighbours
        self.knn_dist = knn_dist 
        self.knn_return_median = knn_return_median
        self.num_triplets = num_triplets

    def _return_trustworthiness(self,X_org:np.array,X_emb:np.array):
        return trustworthiness(X_org,X_emb,n_neighbors=self.n_neighbours,metric = self.metric)
    
    
    @staticmethod
    def _create_knn_search(X,k):
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X.astype(np.float32))
        D,I = index.search(X.astype(np.float32), k)
        return D,I
    
    def _return_knn_overlap(self,X_org:np.array,X_emb:np.array):
        D,I = self._create_knn_search(X_org,self.n_neighbours)
        D_emb,I_emb = self._create_knn_search(X_emb,self.n_neighbours)
        ds_arry = np.zeros(I.shape[0]) 
        dist = None
        if self.knn_dist == 'jaccard':
            dist = jaccard
        elif self.knn_dist == 'hamming':
            dist = hamming
        else:
            raise Exception(f"{self.knn_dist} is not a recognised distance measure for KNN overlap. Please use 'jaccard' or 'hamming'")
        for i in range(I.shape[0]):
            ds_arry[i] = dist(I[i,:],I_emb[i,:])
        return np.median(ds_arry) if self.knn_return_median else ds_arry
  

    def _return_distance_correlation(self,X_org:np.array,X_emb:np.array):
        dist_before = sc.spatial.distance.pdist(X_org,metric = self.metric)
        dist_after = sc.spatial.distance.pdist(X_emb,metric = self.metric)
        return dcor.distance_correlation(dist_before,dist_after)
            
    
    def random_triplet_eval(self,X_org:np.array, X_emb:np.array):
        '''
        Author from Haiyang Huang https://github.com/hyhuang00/scRNA-DR2020/blob/main/experiments/run_eval.py

        This is a function that is used to evaluate the lower dimension embedding.
        An triplet satisfaction score is calculated by evaluating how many randomly
        selected triplets have been violated. Each point will generate 5 triplets.
        Input:
            X: A numpy array with the shape [N, p]. The higher dimension embedding
            of some dataset. Expected to have some clusters.
            X_new: A numpy array with the shape [N, k]. The lower dimension embedding
                of some dataset. Expected to have some clusters as well.
            y: A numpy array with the shape [N, 1]. The labels of the original
            dataset. Used to identify clusters
        Output:
            acc: The score generated by the algorithm.
        '''    
        # Sampling Triplets
        # Five triplet per point
        anchors = np.arange(X_org.shape[0])
        rng = default_rng()
        triplets = rng.choice(anchors, (X_org.shape[0], self.num_triplets, 2))
        triplet_labels = np.zeros((X_org.shape[0], self.num_triplets))
        anchors = anchors.reshape((-1, 1, 1))
        
        # Calculate the distances and generate labels
        b = np.broadcast(anchors, triplets)
        distances = np.empty(b.shape)
        distances.flat = [np.linalg.norm(X_org[u] - X_org[v]) for (u,v) in b]
        labels = distances[:, :, 0] < distances[: , :, 1]
        
        # Calculate distances for LD
        b = np.broadcast(anchors, triplets)
        distances_l = np.empty(b.shape)
        distances_l.flat = [np.linalg.norm(X_emb[u] - X_emb[v]) for (u,v) in b]
        pred_vals = distances_l[:, :, 0] < distances_l[:, :, 1]

        # Compare the labels and return the accuracy
        correct = np.sum(pred_vals == labels)
        acc = correct/X_org.shape[0]/self.num_triplets
        return acc
        

    def score(self, X_org:np.array,X_emb:np.array, subsampling:int=1000, num_iter:int = 10, return_results:bool = False):
        evaluators = {'Trustworthiness': self._return_trustworthiness,
                    'Knn overlap': self._return_knn_overlap,
                    'Distance correlation': self._return_distance_correlation,
                    'Random triplets' : self.random_triplet_eval}
        final_results=score_subsampling(X_org,X_emb,evaluators=evaluators, size=subsampling, num_iter=num_iter, verbose=True, return_dict=True)
        if(return_results):
            return final_results



def metrics_scores_iter(x:np.array, output:np.array, evaluators:dict ,
                        verbose:bool=True, return_dict:bool = False):
    """
    Calculates scores for embedder using different metrics (evaluators). 
    Parameters
    ---------
    x: np.array
        original unemedded data 
    output: np.array
        output array from the embedder for evaluation
    evaluators: dict
        further arguments to include the metrics (as function statement in a dict)
        if the functions take x and output as arguments.
    verbose: bool, optional
        whether to print the results.
    Returns
    ---------
    results: dict
        dictonary of metrics and their calculated scores.
    """
    results = {}
    for name, metric in evaluators.items():
        results.update({name: metric(x, output)})
    if(verbose):
        print_metric_scores(results)
    if(return_dict):
        return results


def print_metric_scores(results:dict):
    """
    Print mertic scores in a table format.
    Parameters
    ---------
    results: dict
            dictionary of results to be printed in a table format. 
    """
    tab = []
    for metric, score in results.items():
        tab.append([metric, score])
    print(tabulate(tab, headers = ["Metric","Score"]))
    print("\n")


def print_means_metric_scores(results:dict):
    """
    Print means of metric scores in a table format with standard deviation.
    Parameters
    ---------
    results: dict
            dictionary of results to be printed in a table format. 
    """
    tab = []
    for metric, score in results.items():
        tab.append([metric, score[0][0], score[0][1]])
    print(tabulate(tab, headers = ["Metric","Mean", "Standard deviation"]))
    print("\n")


def score_subsampling(X:np.array,output:np.array, evaluators:dict, size:int=1000, 
                    num_iter:int = 10,verbose:bool=True, return_dict:bool = False):
    method_start = time.time()

    output_results = defaultdict(list)
    for _ in range(num_iter):
        sample = np.random.choice(np.arange(len(X)),size = size)
        X_org_sub = X[sample,:]
        X_emb_sub = output[sample,:]
        results = metrics_scores_iter(X_org_sub,X_emb_sub,evaluators, verbose=False, return_dict=True)
        for name, score in results.items():
            output_results[name].append(score)

    final_results = defaultdict(list)
    for name, scores in output_results.items():
        final_results[name].append([np.mean(scores), np.std(scores)])
    
    if(verbose):
        print_means_metric_scores(final_results)
        print(f"Supsampling of {size} samples for {num_iter} rounds each")
        print(f"Time taken {round((time.time()-method_start)/60, 2)} minutes")
    
    if(return_dict):
        return final_results

