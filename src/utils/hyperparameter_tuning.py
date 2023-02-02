import random
import numpy as np
import pandas as pd
import time
from collections import defaultdict
import math
import itertools
from hembedder.utils.quality_metrics import metrics_scores_iter
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
            sample_size=10000,
            metric_chuck_size:int=5000,
            standardised:bool = False, 
            num_iter:int = 10,
            n_parjobs:int = 10,
            dtype = np.float32,
            random_state = None,
            **kwargs):
        """
        Setting up parameters for the hyperpameter tuning. Two choices to choose from: random search or grid search.
        Can also benchmark different subsample sizes. To benchmark different subsample sizes, add a list of sizes 
        to sample_size e.g sample_size = [1000,2000] instead of an int.
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
        sample_size:int/list
            number of samples or list of sample size to size down to for embedding. Must be less than
            X size.
        metric_chuck_size:int
            number of samples to size down to for metric calculations. Must not be more than sample_size
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
        self.param_grid = param_grid
        self.sample_size=[self.sample_size] if type(self.sample_size) == int\
                        else self.sample_size
        #Columns for the final result dataframe
        self.record_cols = ['iteration', 'params', 'duration_in_second','sample_size']
        self.result_cols = self.record_cols.copy()
        self.result_cols.extend(list(self.param_grid.keys()))
        self.result_cols.extend(self.evaluators.keys())
        self.var_word = '_variance'
        self.result_cols.extend([s+self.var_word for s in list(self.evaluators.keys())])
        #update sample_size to param_grid once all the columns are added
        self.param_grid.update({'sample_size': self.sample_size}) 
        #result dataframe to be initialised in the search method
        self.results = None
    

    def random_search(self, max_evals:int=300):
        """Random hyperparameter optimization for embedding algorithm. Measuring the results using multiple evaluators
        Adapted from Will Koehrsen for randomized search of hyperpameters for embedding algorithm. Adapted from Will Koehrsen.
        
        Parameters
        ---------
        max_evals: optional, int
            max number of evaluation to seach for the ideal hyperparameter.
        """
        method_start = time.time()
        #setting random state
        random.seed(self.random_state)
        # Dataframe for results
        self.max_evals = max_evals
        print(f"Total number of embedding runs :  {self.max_evals} (combos)x{self.num_iter}(iterations) with",
            f"{self.sample_size} sample_size for the embedding. Exceptions will be raised, if sample",
            f"size is larger than data size which is {self.X.shape[0]}")
        self.results = pd.DataFrame(columns = self.result_cols,
                                    index = list(range(self.max_evals)))                      
        #https://stackoverflow.com/questions/20347766/pythonically-add-header-to-a-csv-file
        with open(self.file_name, 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(self.result_cols)
        # Keep searching until reach max evaluations
        for i in tqdm(range(self.max_evals)):
            #picked  hyperparameters
            hyperparameters = {k: random.sample(v, 1)[0] for k, v in self.param_grid.items()}
            # Store parameter values in results frame
            for k,v in hyperparameters.items():self.results.at[i,k] = v
            hyperparameters.update(self.kwargs)
            #Embedding data for each hyperparameter set per num_iter
            embedded_data, times = self.get_embedded_data(hyperparameters)
            #Calculating performance for the embedded data using thread pool
            pool = ThreadPool(10)
            scores=pool.map(self.get_scores, embedded_data)
            pool.close()
            pool.join()
            self.store_param_results(i,scores, hyperparameters, times)
        self.print_final_results()
        print("Finish tuning in ",round((time.time() - method_start)/60, 2), "minutes.")
  

    def grid_search(self):
        """Grid hyperparameter optimization for embedding algorithm. Measuring the results using multiple evaluators
        Adapted from Will Koehrsen for grid searching of hyperpameters for embedding algorithm. Adapted from Will Koehrsen.
        """
        # https://codereview.stackexchange.com/questions/171173/list-all-possible-permutations-from-a-python-dictionary-of-lists
        keys, values = zip(*self.param_grid.items())
        # Keep the length of all set of the parameter combinations
        param_len = 1
        for v in values:param_len=param_len*len(v)
        print(f"Total number of embedding runs :  {param_len} (combos)x{self.num_iter}(iterations) with",
            f"{self.sample_size} sample_size for the embedding. Exceptions will be raised, if sample",
            f" size is larger than data size which is {self.X.shape[0]}")
        param_len-=1
        method_start = time.time()
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
        with tqdm(total=self.max_evals) as pbar:
            for i in itertools.product(*values):
                # Retrieving the parameter set for a given value i
                hyperparameters = dict(zip(keys, i))
                # dictionary for storing average results 
                for k,v in hyperparameters.items():self.results.at[counter,k] = v 
                hyperparameters.update(self.kwargs)
                #Embedding data for each hyperparameter setting per num_iter
                embedded_data, times = self.get_embedded_data(hyperparameters)
                #Calculating performance for the embedded data using thread pool
                pool = ThreadPool(self.n_parjobs)
                scores=pool.map(self.get_scores, (x for x in embedded_data))
                pool.close()
                pool.join()
                self.store_param_results(counter,scores, hyperparameters, times)
                counter+=1
                pbar.update(1)
        self.print_final_results()
        print("Finish tuning in ",round((time.time() - method_start)/60, 2), "minutes.")


    def get_embedded_data(self,parameter:dict):
        """(HELPER) Get embedded data from original data by subsampling and 
        place them in a big list of dictionaries for further benchmarking

        Parameter
        ----------
        paramter:dict
            paramters for the embedder to be tuned
        
        Return
        ---------
        benchmark_list:list
            big list of dictionaries containing the orignal data, the embedded data, 
            and the evaluator list for further benchmarking. The number of sets of the orignal data 
            and its corresponding embedded data depends on the number of iters (num_iter). 
            The size of the data in the benchmark_list can be reduced or increased by the 
            metric_chuck_size argument. 
        times: list
            contains the time it takes to run once iteration of the tuning.
        """
        benchmark_list = []
        times = []
        #paramter contains sample_size so as to vary 
        # the sample size when tuning
        size = parameter['sample_size']
        del parameter['sample_size']
        if(size > self.X.shape[0]): raise Exception(f"{size} sample_size is larger than dataset size"+
        f" ({self.X.shape[0]}). Please reduce sample_size to equal to or less than {self.X.shape[0]}" )
        if(self.metric_chuck_size > size): raise Exception(f"{self.metric_chuck_size} metric_chuck_size is"+
        f" larger than sample_size ({size}). Please reduce metric_chuck_size to equal to or less than {size}" )
        for iter in range(self.num_iter):
            start = time.time()
            #Get indexes for subsampling of data for embedding
            indexes_embedder = subsampling_return_indexes(self.X,size)
            # Evaluate randomly selected hyperparameters
            sub_original = self.X[indexes_embedder]
            embedding_data = sub_original.copy().astype(self.dtype)
            #Check for scaling
            if(self.standardised):
                scaler = StandardScaler()
                embedding_data = scaler.fit_transform(sub_original).astype(self.dtype)
            #Get indexes for subsampling of data for benchmarking
            indexes_metrics= subsampling_return_indexes(sub_original, 
                                                self.metric_chuck_size)
            # Create a dictionary for later reference in multi-thread
            emb_dict = {"original" : sub_original[indexes_metrics],
                    "embedded" : self.embedder(**parameter).
                    fit_transform(embedding_data).
                    astype(self.dtype)[indexes_metrics],
                    "evaluators":self.evaluators}
            benchmark_list.append(emb_dict)
            times.append(time.time()-start)
        return benchmark_list, times
        
    @staticmethod
    def get_scores(benchmark_dict): 
        """(HELPER)Get performance metrics for the orignal data and the embedded data 

        Parameter
        ---------
        benchmark_dict:dict
            Dictionaries containing the orignal data, the embedded data, 
            and the evaluator list for further benchmarking.
        
        Return
        ---------
            the benchmark scores
        """
        scores = metrics_scores_iter(
            benchmark_dict["original"],
            benchmark_dict["embedded"],
            benchmark_dict["evaluators"],
            return_dict= True, verbose=False)
        return scores
       
    def store_param_results(self,indx,scores, hyperparameters, times):
        """(HELPER) Store the results of the tuning in dataframe and on disk.
        
        Parameter
        --------
        indx:
            position in the dataframe and disk to store the values
        scores:
            list of all the scores to be stored
        hyperparameters:
            dictionary of embedder parameters to be stored
        times:
            list of times taken for the embedder to finish. 
        """
        scores_dict = defaultdict(list)
        for s in scores:
            for k, v in s.items():scores_dict[k].append(v)  
        for metric, score_ls in scores_dict.items():
            self.results.at[indx, metric]= np.mean(score_ls)
            self.results.at[indx, metric+self.var_word] = np.std(score_ls)
        self.results.iloc[indx, :len(self.record_cols)-1] = \
            [indx,hyperparameters,np.mean(times)]
        with open(self.file_name,'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(self.results.iloc[indx,:].values))
    
    def print_final_results(self):
        """Sort with best score on top and print the final best results"""
        self.results.sort_values(list(self.evaluators.keys()), 
        ascending = self.ascending, inplace = True)
        self.results.reset_index(inplace = True)
        print(f"Hyperpameter tuning is done and the best scores with",
        f"{int(self.results.loc[0]['sample_size'])} sample size are")
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