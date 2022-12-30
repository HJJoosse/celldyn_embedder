import random
import numpy as np
import pandas as pd
import time
from random import randint
from collections import defaultdict
import math
import itertools
from utils.quality_metrics import *




__author__ = "Chontira Chumsaeng but adapted from Will Koehrsen"

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


def random_seach(X, embedder, evaluators:dict, param_grid:dict ,ascending:list,subsampling = False,seperating_param_values:bool =False, max_evals:int = 300, num_iter:int = 5, random_state = None, **kwargs):
    """
    Random hyperparameter optimization for embedding algorithm. Measuring the results using multiple evaluators
    Adapted from Will Koehrsen for randomized search of hyperpameters for embedding algorithm.

    Paramters
    ---------
    X: numpy array or dataframe
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
    subsampling: optional, bool or int 
        needs to be an int if subsampling is to be used with a certain number of rows, otherwise the full dataset
        will be used.
    seperating_param_values: optional, bool
        whether to seperate the parameter result column into seperated columns 
    max_evals: optional, int
        max number of evaluation to seach for the ideal hyperparameter.
    num_iters: optional, int
        number of iterations to run for each hyperparameter setting.
    random_state: optional, int
        set to None. If int is used e.g. 0, the randomness is deterministic
    **kwargs:
        Whatever key words argument that are allow in the embedder 
    
    Returns
    ---------
    results: DataFrame
        containing the scores of the evaluations, the time it takes to run the embedder on average (in second),
        the hyperparameter settings, and order with which hyperparameter was ran. 
    """
    X = np.asarray(X)
    method_start = time.time()
    precent_range = list(range(10,110,10))
    #setting random state
    random.seed(random_state)
    # Dataframe for values other then the results
    records = pd.DataFrame(columns = ['iteration', 'params', 'duration_in_second'],
                                  index = list(range(max_evals)))
    # Dataframe for results
    results = pd.DataFrame(columns = evaluators.keys(),
                                  index = list(range(max_evals)))
    std_words = '_variance'
    std_results = pd.DataFrame(columns = [s+std_words for s in list(evaluators.keys())],
                            index= list(range(max_evals)))
    
    # Keep searching until reach max evaluations
    for i in range(max_evals):
        #picked  hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        hyperparameters.update(kwargs)
        # dictionary for storing average results
        num_labels = [] 
        scores = defaultdict(list)
        times = []

        #Run evaluation for each hyperparameter setting per num_iter
        for num in range(num_iter):
            sub = X
            start = time.time()
            if(type(subsampling) == int):
                sub = numpy_sampling(sub, subsampling)
            # Evaluate randomly selected hyperparameters
            embedded = embedder(**hyperparameters).fit_transform(sub)
            times.append(time.time()-start)
            metric_results = metrics_scores_iter(sub, embedded, evaluators, return_dict= True)
            for metric, score in metric_results.items():
                scores[metric].append(score)
        for metric, score_ls in scores.items():
            results.at[i, metric]= np.mean(score_ls)
            std_results.at[i, metric+std_words] = np.std(score_ls)
            
        records.loc[i, :] = [i,hyperparameters,np.mean(times)]
        #Print the percentage until finish
        percent_done = roundup(int((i/max_evals)*100))
        if(percent_done in precent_range):
            print(f"Around {percent_done}% done." )
            precent_range.remove(percent_done)
    #combine all three of the tables for sco
    records  = pd.concat([records, results, std_results], axis=1)
    if(seperating_param_values):
        records  = seperate_param_values(records)

    # Sort with best score on top
    records.sort_values(list(evaluators.keys()), ascending = ascending, inplace = True)
    records.reset_index(inplace = True)

    print("Hyperpameter tuning is done and the best scores are:")
    print(records.loc[0][list(evaluators.keys())]) 
    print("with parameter:", records['params'][0])
    print("Finish tuning in ",round((time.time() - method_start)/60, 2), "minutes.")

    return records 



def grid_seach(X, embedder, evaluators:dict, param_grid:dict ,ascending:list,subsampling = False, seperating_param_values:bool =False,num_iter:int = 5, random_state = None, **kwargs):
    """
    Grid hyperparameter optimization for embedding algorithm. Measuring the results using multiple evaluators
    Adapted from Will Koehrsen for grid searching of hyperpameters for embedding algorithm.
    Paramters
    ---------
    X: numpy array or dataframe
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
    subsampling: optional, bool or int 
        needs to be an int if subsampling is to be used with a certain number of rows, otherwise the full dataset
        will be used.
    seperating_param_values: optional, bool
        whether to seperate the parameter result column into seperated columns 
    num_iters: optional, int
        number of iterations to run for each hyperparameter setting.
    random_state: optional, int
        set to None. If int is used e.g. 0, the randomness is deterministic.
    **kwargs:
        Whatever key words that are allow in the embedder
    Returns
    ---------
    results: DataFrame
        containing the scores of the evaluations, the time it takes to run the embedder on average (in second),
        the hyperparameter settings, and order with which hyperparameter was ran.
        
    """
    X = np.asarray(X)
    # https://codereview.stackexchange.com/questions/171173/list-all-possible-permutations-from-a-python-dictionary-of-lists
    keys, values = zip(*param_grid.items())
    # Keep the length of all set of the parameter combinations
    param_len = 1
    for v in values:
        param_len=param_len*len(v)
    param_len-=1
    method_start = time.time()
    precent_range = list(range(10,110,10))
    #setting random state
    random.seed(random_state)
    # Dataframe for recording things other then the results
    records = pd.DataFrame(columns = ['iteration', 'params', 'duration_in_second'],
                                index = list(range(param_len)))
    # Dataframe for results
    results = pd.DataFrame(columns = evaluators.keys(),
                                index = list(range(param_len)))
    std_words = '_variance'
    std_results = pd.DataFrame(columns = [s+std_words for s in list(evaluators.keys())],
                            index= list(range(param_len)))
    counter = 0

    for i in itertools.product(*values):
         # Retrieving the parameter set for a given value i
        hyperparameters = dict(zip(keys, i))
        hyperparameters.update(kwargs)
        # dictionary for storing average results 
        num_labels = [] 
        scores = defaultdict(list)
        times = []
        #Run evaluation for each hyperparameter setting per num_iter
        for num in range(num_iter):
            sub = X
            start = time.time()
            if(type(subsampling) == int):
                sub = numpy_sampling(sub, subsampling)
            # Evaluate randomly selected hyperparameters
            embedded = embedder(**hyperparameters).fit_transform(sub)            
            times.append(time.time()-start)
            metric_results = metrics_scores_iter(sub, embedded, evaluators, return_dict= True)
            for metric, score in metric_results.items():
                scores[metric].append(score)
        for metric, score_ls in scores.items():
            results.at[counter, metric]= np.mean(score_ls)
            std_results.at[counter, metric+std_words] = np.std(score_ls)
        records.loc[counter, :] = [counter,hyperparameters,np.mean(times)]
        #Print the percentage until finish
        percent_done = roundup(int((counter/param_len)*100))
        if(percent_done in precent_range):
            print(f"Around {percent_done}% done." )
            precent_range.remove(percent_done)
        counter+=1

    records  = pd.concat([records, results, std_results], axis=1)
    if(seperating_param_values):
        records  = seperate_param_values(records)

    # Sort with best score on top
    records.sort_values(list(evaluators.keys()), ascending = ascending, inplace = True)
    records.reset_index(inplace = True)

    print("Hyperpameter tuning is done and the best scores are:")
    print(records.loc[0][list(evaluators.keys())])
    print("with parameter:", records['params'][0])
    print("Finish tuning in ",round((time.time() - method_start)/60, 2), "minutes.")

    return records 


def seperate_param_values(record_frame):
    """
    HELPER FUNCTION for seperating parameter values into indiviudal columns 
    Paramters
    ---------
    record_frame: dataframe
        the record dataframe for seperating of the parameter values into different columns.
    Returns
    ---------
    results: DataFrame
        new dataframe containing the original record frame with the seperated paramter columns attached to it.
        
    """
    results_len = len(record_frame)
    colnames = record_frame["params"].loc[0].keys()
    param_values = pd.DataFrame(columns = colnames, index= list(range(results_len)))
    for i in range(len(record_frame)):
        param = record_frame["params"].loc[i]
    
        for k,v in param.items():
            param_values.at[i,k] = v
    records  = pd.concat([record_frame, param_values], axis=1)
    return records