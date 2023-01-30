__author__ = "Chontira Chumsaeng but adapted from Will Koehrsen"
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
            subsampling:int,
            standardised:bool = False, 
            num_iter:int = 10,
            n_parjobs:int = 10,
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
        subsampling:int 
            number of samples to subsample to
        standardised:optional, bool
            whether to standardise the data
        num_iters: optional, int
            number of iterations to run for each hyperparameter setting.
        random_state: optional, int
            set to None. If int is used e.g. 0, the randomness is deterministic
        **kwargs:
            Whatever key words argument that are allowed in the embedder 
        """
        self.X = np.asarray(X, dtype=np.float16)
        self.embedder = embedder
        self.evaluators = evaluators 
        self.param_grid = param_grid
        self.ascending = ascending
        self.file_name = file_name
        self.subsampling = subsampling
        self.standardised = standardised
        self.max_evals = None
        self.num_iter = num_iter
        self.n_parjobs = n_parjobs
        self.random_state = random_state
        self.kwargs = kwargs
        self.percent_range = None
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
            # dictionary for storing average results
            # Store parameter values in results frame
            for k,v in hyperparameters.items():
                self.results.at[i,k] = v
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
        for i in itertools.product(*values):
            # Retrieving the parameter set for a given value i
            hyperparameters = dict(zip(keys, i))
            hyperparameters.update(self.kwargs)
            # dictionary for storing average results 
            for k,v in hyperparameters.items():
                self.results.at[counter,k] = v 
            #Embedding data for each hyperparameter setting per num_iter
            embedded_data, times = self.get_embedded_data(hyperparameters)
            #Calculating performance for the embedded data using thread pool
            pool = ThreadPool(self.n_parjobs)
            scores=pool.map(self.get_scores, (x for x in embedded_data))
            pool.close()
            pool.join()
            self.store_param_results(counter,scores, hyperparameters, times)
            self.print_percentage_done(counter)
            counter+=1
        self.print_final_results()
        print("Finish tuning in ",round((time.time() - method_start)/60, 2), "minutes.")

    def get_embedded_data(self,parameter:dict):
        #Get embedded data from original data by subsampling
        embedded_data = []
        times = []
        for iter in range(self.num_iter):
            start = time.time()
            sub = numpy_sampling(self.X, self.subsampling)
            # Evaluate randomly selected hyperparameters
            CD_scaled = sub.copy()
            if(self.standardised):
                scaler = StandardScaler()
                CD_scaled = scaler.fit_transform(sub)
            # Create a dictionary for later reference in multi-thread
            emb_dict = {"original" : sub,
                        "embedded" : self.embedder(**parameter).fit_transform(sub).astype(np.float16),
                        "evaluators":self.evaluators}
            embedded_data.append(emb_dict)
            times.append(time.time()-start)
        return embedded_data, times
        
    @staticmethod
    def get_scores(embedded_info): 
        #Get performance metrics for each subsampled embedder
        scores = metrics_scores_iter(
            embedded_info["original"],
            embedded_info["embedded"],
            embedded_info["evaluators"],
            return_dict= True, verbose=False)
        return scores
       
    def store_param_results(self,indx,scores, hyperparameters, times):
        #Store the results in dataframe and on disk
        scores_dict = defaultdict(list)
        for s in scores:
            for k, v in s.items():
                scores_dict[k].append(v)  
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
        print("Hyperpameter tuning is done and the best scores are:")
        print(self.results.loc[0][list(self.evaluators.keys())]) 
        print("with parameter:", self.results['params'][0])
       
    
def numpy_sampling(X, subsampling):  
    n_data = len(X) 
    idx = np.arange(n_data) 
    np.random.shuffle(idx) 
    return X[idx[: subsampling],:] 

def roundup(x):
    """
    From https://stackoverflow.com/questions/26454649/python-round-up-to-the-nearest-ten
    """
    return int(math.ceil(x / 10.0)) * 10