# trustworthiness --> sklearn
# distance correlation correlation --> zelf chefffen
# knn-overlap - distance curve and integral
# poincarre


### knn_overlap -> find knn in true space random sample. Then embedding do the same for indices -> jaccard score
__author__ = "Bram van ES","Huibert-Jan Joosse","Chontira Chumsaeng"

from scipy.spatial.distance import jaccard, hamming
import numpy as np
from numpy.random import default_rng
from sklearn.manifold import trustworthiness
import dcor
import faiss
import time
from tabulate import tabulate
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors


class CDEmbeddingPerformance:
    """
    Class for calulating the embedding quality. Metrics include trustworthiness, Knn overlap, distance correlation, and random triplet scores 
    """

    def __init__(self,metric='euclidean',n_neighbours:int=15, knn_dist:str='jaccard', num_triplets:int=5):
        """
        Setting up parameters for the quality metircs
        Paramters
        ---------
        metric: string or function
            distance metric for trustworiness and distance correlation
        n_neighbours:int
            number of neighbours for trustworiness and knn overlap scores
        knn_dist:string
            distance metric for calculating overlap between neighbours in knn overlap. 'hamming' or 'jaccard'
        num_triplets:int
            paramter for random triplets calculation.
        """
        self.metric = metric
        self.n_neighbours = n_neighbours
        self.knn_dist = knn_dist 
        self.num_triplets = num_triplets

    def _return_trustworthiness(self,X_org:np.array,X_emb:np.array):
        """
        Function for returning trustworthiness score from sklearn.manifold.
        Parameters
        ----------
        X_org:np.array
            the original dataset as np.array
        X_emb:np.array
            the embedded data as np.array
        Returns
        -----------
        Trustworithiness score between 0 and 1. Higher means better
        """
        return trustworthiness(X_org,X_emb,n_neighbors=self.n_neighbours,metric = self.metric)
    
    
    @staticmethod
    def _create_knn_search(X,k):
        """
        HELPER function for knn_overlap
        """
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X.astype(np.float32))
        D,I = index.search(X.astype(np.float32), k)
        return D,I
    
    def _return_knn_overlap(self,X_org:np.array,X_emb:np.array, knn_return_median:bool = True):
        """
        Function for returning nearest neighbourhood overlap score. Overlap between the high dimension and low dimension data
        Parameters
        ----------
        X_org:np.array
            the original dataset as np.array
        X_emb:np.array
            the embedded data as np.array
        knn_return_median:bool
            whether to return the median of the knn overlap scores. This should be true if knn is to be used with other methods here.
        Returns
        -----------
        knn overlap score between 0 and 1. Lower means better
        """
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
        return np.median(ds_arry) if knn_return_median else ds_arry
  

    def _return_distance_correlation(self,X_org:np.array,X_emb:np.array):
        """
        Function for returning distance correlation from dcor between the high dimension and low dimension data
        Parameters
        ----------
        X_org:np.array
            the original dataset as np.array
        X_emb:np.array
            the embedded data as np.array
        Returns
        -----------
        distance correlation score between 0 and 1. Higher means better
        """
        return dcor.distance_correlation(X_org,X_emb)
            
    
    @staticmethod
    def calculate_distance_random_triplets(X:np.array,anchors, triplets):
        """
        HELPER function for random_triplet_eval to calculate distance for X and generate the labels
        """
        b = np.broadcast(anchors, triplets)
        distances = np.empty(b.shape)
        distances.flat = [np.linalg.norm(X[u] - X[v]) for (u,v) in b]
        labels = distances[:, :, 0] < distances[: , :, 1]
        return labels
    
    @staticmethod
    def calculate_anchors_and_triplets(X:np.array, num_triplets:int):
        """
        HELPER function for random_triplet_eval to create the achors and triplets for evaluating the
        triplets violation
        """
        anchors = np.arange(X.shape[0])
        rng = default_rng()
        triplets = rng.choice(anchors, (X.shape[0], num_triplets, 2))
        triplet_labels = np.zeros((X.shape[0], num_triplets))
        anchors = anchors.reshape((-1, 1, 1))
        return anchors,triplets
            
    def random_triplet_eval(self,X_org:np.array, X_emb:np.array):
        '''
        Author: Haiyang Huang https://github.com/hyhuang00/scRNA-DR2020/blob/main/experiments/run_eval.py
        This is a function that is used to evaluate the lower dimension embedding.
        An triplet satisfaction score is calculated by evaluating how many randomly
        selected triplets have been violated. Each point will generate 5 triplets.
        Parameters
        ----------
            X_org: A numpy array with the shape [N, p]. The higher dimension embedding
            of some dataset. Expected to have some clusters.
            X_emb: A numpy array with the shape [N, k]. The lower dimension embedding
                of some dataset. Expected to have some clusters as well.
        Returns
        ----------
            acc: The score generated by the algorithm.
        '''    
        # Sampling Triplets
        # Five triplet per point
        anchors,triplets = self.calculate_anchors_and_triplets(X_org,self.num_triplets)
        
        # Calculate the distances and generate labels
        labels = self.calculate_distance_random_triplets(X_org,anchors, triplets)
        
        # Calculate distances for LD
        pred_vals = self.calculate_distance_random_triplets(X_emb,anchors, triplets)

        # Compare the labels and return the accuracy
        correct = np.sum(pred_vals == labels)
        acc = correct/X_org.shape[0]/self.num_triplets
        return acc
        
    
    def neighbor_kept_ratio_eval(self,X_org:np.array, X_emb:np.array):
        '''
        Author: Haiyang Huang https://github.com/hyhuang00/scRNA-DR2020/blob/main/experiments/run_eval.py
        This is a function that evaluates the local structure preservation.
        A nearest neighbor set is constructed on both the high dimensional space and
        the low dimensional space.
        Input:
            X_org: A numpy array with the shape [N, p]. The higher dimension embedding
            of some dataset. Expected to have some clusters.
            X_emb: A numpy array with the shape [N, k]. The lower dimension embedding
                of some dataset. Expected to have some clusters as well.
        Output:
            acc: The score generated by the algorithm.
        '''
        nn_hd = NearestNeighbors(n_neighbors=self.n_neighbours+1)
        nn_ld = NearestNeighbors(n_neighbors=self.n_neighbours+1)
        nn_hd.fit(X_org)
        nn_ld.fit(X_emb)
        # Construct a k-neighbors graph, where 1 indicates a neighbor relationship
        # and 0 means otherwise, resulting in a graph of the shape n * n
        graph_hd = nn_hd.kneighbors_graph(X_org).toarray()
        graph_hd -= np.eye(X_org.shape[0]) # Removing diagonal
        graph_ld = nn_ld.kneighbors_graph(X_emb).toarray()
        graph_ld -= np.eye(X_org.shape[0]) # Removing diagonal
        neighbor_kept = np.sum(graph_hd * graph_ld).astype(float)
        neighbor_kept_ratio = neighbor_kept / self.n_neighbours / X_org.shape[0]
        return neighbor_kept_ratio

    def score(self, X_org:np.array,X_emb:np.array, subsampling:int=1000, num_iter:int = 10, return_results:bool = False):
        """
        Return embedding trustworithness, knn overlap, distance correlation, and random triplets scores.
        Parameters
        ----------
        X_org:np.array
            the original dataset as np.array
        X_emb:np.array
            the embedded data as np.array
        subsampling:int
            the number of samples for the data to be subsampled down to
        num_iter:
            the amount of iteration for the algorithms to cycle through for calculating the scores
        return_results:bool
            whether to return the results as dictionary
        Returns
        ----------
        final_results:dict, optional
            the calculated results in a dictionary.

        """
        evaluators = {'Trustworthiness': self._return_trustworthiness,
                    'Knn overlap': self._return_knn_overlap,
                    'Distance correlation': self._return_distance_correlation,
                    'Random triplets' : self.random_triplet_eval,
                    'neighbor kept ratio' : self.neighbor_kept_ratio_eval}
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
    """
    Calculates quality metric scores with subsampling to reduce processing time.
    Parameters
    ---------
    x: np.array
        original unemedded data 
    output: np.array
        output array from the embedder for evaluation
    evaluators: dict
        further arguments to include the metrics (as function statement in a dict)
        if the functions take x and output as arguments.
    size:int
        the number of samples for the data to be subsampled down to
    num_iter:
        the amount of iteration for the algorithms to cycle through for calculating the scores
    verbose: bool, optional
        whether to print the results.
    Returns
    ---------
    results: dict
        dictonary of metrics and their calculated scores.
    """
    method_start = time.time()
    output_results = defaultdict(list)
    for iter in range(num_iter):
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

