import random
import numpy as np
import pandas as pd
import time
from collections import defaultdict
import math
import itertools
from hembedder.utils.quality_metrics import metrics_scores_iter, score_subsampling
import csv
from sklearn.preprocessing import StandardScaler
from multiprocessing.dummy import Pool as ThreadPool

from tqdm import tqdm

class Hyperparameter_tuning:
    """
    Class for hyperparameter tuning of embedders.
    """

    def __init__(
            self,X:np.array,
            embedder,
            evaluators:dict,
            param_grid:dict,
            ascending:list,
            file_name:str,
            sample_size,
            metric_chuck_size:int=5000,
            standardised:bool = False, 
            num_iter:int = 10,
            n_parjobs:int = 10,
            dtype = np.float32,
            random_state = None,
            **kwargs):
        """
        Setting up parameters for the hyperpameter tuning. Two choices to choose from: random search or grid search.

        Paramters
        ---------
        X: numpy array
            the data that needs to be fitted to the embedder.
        embedder: function
            the embedder that needs to be hyperparameter tunned.
        evaluators: dict
            the dictionary that stores all the evaluators for measuring the performace of the embedded data
        param_grid: dict
            dictionary of the embedder parameters with range of values to be seleted by the search.
        ascending: list of bool
            list length equals to the number of evaluator
            whether the scores should be displayed in ascending order or descending order. True = ascending
        file_name:string
            name of the csv file for saving of the results on disc
        subsampling:int/list
            number of samples or list of sample size to size down to for embedding
        metric_chuck_size:int
            number of samples to size down to for metric calculations. Must not be more than subsampling
        standardised:optional, bool
            whether to standardise the data
        num_iters: optional, int
            number of iterations to run for each hyperparameter setting.
        dtype:numpy array of floats,
            X's datatype to be converted/or use during tuning. If the dataset is too large use np.float32 or lower.
        random_state: optional, int
            set to None. If int is used e.g. 0, the randomness is deterministic
        **kwargs:
            Whatever key words argument that are allowed in the embedder 
        """
        self.dtype = dtype
        self.X = np.asarray(X, dtype=self.dtype)
        self.embedder = embedder
        self.evaluators = evaluators 
        self.ascending = ascending
        self.file_name = file_name
        self.sample_size = sample_size
        self.metric_chuck_size = metric_chuck_size
        self.standardised = standardised
        self.max_evals = None
        self.num_iter = num_iter
        self.n_parjobs = n_parjobs
        self.random_state = random_state
        self.kwargs = kwargs
        self.percent_range = None
        self.param_grid = param_grid
        self.sample_size=[self.sample_size] if type(self.sample_size) == int else self.sample_size
        self.param_grid.update({'sample_size': self.sample_size}) 
        #Columns for the final result dataframe
        self.var_word = '_variance'
        self.record_cols = ['iteration', 'params', 'duration_in_second']
        self.result_cols = self.record_cols.copy()
        self.result_cols.extend(self.evaluators.keys())
        self.result_cols.extend([s+self.var_word for s in list(self.evaluators.keys())])
        self.result_cols.extend(list(self.param_grid.keys()))
        self.result_cols.extend(list(self.kwargs.keys()))
        #result dataframe to be initialised in the search method
        self.results = None
    

    def random_search(self, max_evals:int=300):
        """
        Random hyperparameter optimization for embedding algorithm. Measuring the results using multiple evaluators
        Adapted from Will Koehrsen for randomized search of hyperpameters for embedding algorithm.
        
        Parameters
        ---------
        max_evals: optional, int
            max number of evaluation to seach for the ideal hyperparameter.
        """
        method_start = time.time()
        self.percent_range = list(range(10,110,10))
        #setting random state
        random.seed(self.random_state)
        # Dataframe for results
        self.max_evals = max_evals
        self.results = pd.DataFrame(columns = self.result_cols,
                                    index = list(range(self.max_evals)))                      
        #https://stackoverflow.com/questions/20347766/pythonically-add-header-to-a-csv-file
        with open(self.file_name, 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(self.result_cols)
        # Keep searching until reach max evaluations
        for i in range(self.max_evals):
            #picked  hyperparameters
            hyperparameters = {k: random.sample(v, 1)[0] for k, v in self.param_grid.items()}
            hyperparameters.update(self.kwargs)
            # Store parameter values in results frame
            for k,v in hyperparameters.items():self.results.at[i,k] = v
            scores = defaultdict(list)
            times = []
            #Embedding data for each hyperparameter setting per num_iter
            embedded_data, times = self.get_embedded_data(hyperparameters)
            #Calculating performance for the embedded data using thread pool
            pool = ThreadPool(10)
            scores=pool.map(self.get_scores, embedded_data)
            pool.close()
            pool.join()
            self.store_param_results(i,scores, hyperparameters, times)
            self.print_percentage_done(i)
        self.print_final_results()
        print("Finish tuning in ",round((time.time() - method_start)/60, 2), "minutes.")
  

    def grid_search(self):
        """
        Grid hyperparameter optimization for embedding algorithm. Measuring the results using multiple evaluators
        Adapted from Will Koehrsen for grid searching of hyperpameters for embedding algorithm.
        """
        # https://codereview.stackexchange.com/questions/171173/list-all-possible-permutations-from-a-python-dictionary-of-lists
        keys, values = zip(*self.param_grid.items())
        # Keep the length of all set of the parameter combinations
        param_len = 1
        for v in values:
            param_len=param_len*len(v)
        print(f"Total number of embedding runs :  {param_len} (combos)x{self.num_iter}(iterations) \
              with {self.subsampling} samples for the embedding and \
              {self.eval_sampling} samples for the evaluation")

        param_len-=1
        method_start = time.time()
        self.percent_range = list(range(10,110,10))
        #setting random state
        random.seed(self.random_state)
        self.max_evals = param_len
        # Dataframe for results
        self.results = pd.DataFrame(columns = self.result_cols,
                                    index = list(range(self.max_evals)))                      
        #https://stackoverflow.com/questions/20347766/pythonically-add-header-to-a-csv-file
        with open(self.file_name, 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(self.result_cols)
        counter = 0

        self.embedded_times = []
        self.metric_times = []
        for i in tqdm(itertools.product(*values)):
            # Retrieving the parameter set for a given value i
            hyperparameters = dict(zip(keys, i))
            hyperparameters.update(self.kwargs)
            # dictionary for storing average results 
            for k,v in hyperparameters.items():self.results.at[counter,k] = v 
            #Embedding data for each hyperparameter setting per num_iter
            emb_start = time.time()
            embedded_data, times = self.get_embedded_data(hyperparameters)
            self.embedded_times.append(dict(zip(keys, i)).update({'embedding_time': time.time()-emb_start}))

            #Calculating performance for the embedded data using thread pool
            metric_start = time.time()
            pool = ThreadPool(self.n_parjobs)
            scores=pool.map(self.get_scores, (x for x in embedded_data))
            pool.close()
            pool.join()
            self.metric_times.append(dict(zip(keys, i)).update({'metric_time': time.time()-metric_start}))

            self.store_param_results(counter,scores, hyperparameters, times)
            #self.print_percentage_done(counter)
            counter+=1
        self.print_final_results()
        print("Finish tuning in ",round((time.time() - method_start)/60, 2), "minutes.")


    def get_embedded_data(self,parameter:dict):
        #Get embedded data from original data by subsampling
        brenchmark_list = []
        times = []
        size = parameter['sample_size']
        del parameter['sample_size']
        for iter in range(self.num_iter):
            start = time.time()
            indexes_embedder = subsampling_return_indexes(self.X,size)
            # Evaluate randomly selected hyperparameters
            sub_original = self.X[indexes_embedder]
            embedded_data = sub_original.copy().astype(self.dtype)
            if(self.standardised):
                scaler = StandardScaler()
                embedded_data = scaler.fit_transform(sub_original).astype(self.dtype)
            # Create a dictionary for later reference in multi-thread
            indexes_metrics= subsampling_return_indexes(sub_original, self.metric_chuck_size)
            emb_dict = {"original" : sub_original[indexes_metrics],
                        "embedded" : self.embedder(**parameter).fit_transform(embedded_data).astype(self.dtype)[indexes_metrics],
                        "evaluators":self.evaluators}
            brenchmark_list.append(emb_dict)
            times.append(time.time()-start)
        return brenchmark_list, times
        
    @staticmethod
    def get_scores(embedded_info): 
        #Get performance metrics for each subsampled embedder
        #scores = metrics_scores_iter(
        #    embedded_info["original"],
        #    embedded_info["embedded"],
        #    embedded_info["evaluators"],
        #   return_dict= True, verbose=False)
        scores = score_subsampling(
            embedded_info["original"],
            embedded_info["embedded"],
            embedded_info["evaluators"],
            size=1000,
            num_iter=10,
           return_dict= True, verbose=False)
        return scores
       
    def store_param_results(self,indx,scores, hyperparameters, times):
        #Store the results in dataframe and on disk
        scores_dict = defaultdict(list)
        for s in scores:
            for k, v in s.items():scores_dict[k].append(v)  
        for metric, score_ls in scores_dict.items():
            self.results.at[indx, metric]= np.mean(score_ls)
            self.results.at[indx, metric+self.var_word] = np.std(score_ls)
        self.results.iloc[indx, :len(self.record_cols)] = [indx,hyperparameters,np.mean(times)]
        with open(self.file_name,'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(self.results.iloc[indx,:].values))

    def print_percentage_done(self, i):
        #Print the percentage until finish
        percent_done = roundup(int((i/self.max_evals)*100))
        if(percent_done in self.percent_range):
            print(f"Around {percent_done}% done." )
            self.percent_range.remove(percent_done)
    
    def print_final_results(self):
        # Sort with best score on top and print the final best results
        self.results.sort_values(list(self.evaluators.keys()), ascending = self.ascending, inplace = True)
        self.results.reset_index(inplace = True)
        print(f"Hyperpameter tuning is done and the best scores are with {self.results.loc[0]['sample_size']} sample size")
        print(self.results.loc[0][list(self.evaluators.keys())]) 
        print("with parameter:", self.results['params'][0])
       
    
def subsampling_return_indexes(X, subsampling):
    rand = np.random.default_rng()
    n_data = len(X) 
    subsampling = min(n_data, subsampling) 
    return  rand.choice(np.arange(n_data), size=subsampling, replace=False)

def roundup(x):
    """
    From https://stackoverflow.com/questions/26454649/python-round-up-to-the-nearest-ten
    """
    return int(math.ceil(x / 10.0)) * 10