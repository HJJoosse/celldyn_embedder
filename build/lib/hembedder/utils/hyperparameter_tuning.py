import random
import numpy as np
import pandas as pd
import time
from random import randint
from collections import defaultdict
import math
import itertools
from hembedder.utils.quality_metrics import metrics_scores_iter
import csv




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



def random_seach(X, embedder, evaluators:dict, param_grid:dict ,ascending:list,file_name:str,subsampling = False, max_evals:int = 300, num_iter:int = 5, random_state = None, **kwargs):
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
    file_name:string
        name of the csv file for saving of the results on disc
    subsampling: optional, bool or int 
        needs to be an int if subsampling is to be used with a certain number of rows, otherwise the full dataset
        will be used. 
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

    #setting up all the columns to be reference later
    var_word = '_variance'
    record_cols = ['iteration', 'params', 'duration_in_second']
    eval_cols = evaluators.keys()
    var_cols = [s+var_word for s in list(evaluators.keys())]
    param_cols = list(param_grid.keys())
    param_cols.extend(list(kwargs.keys()))
    result_cols = record_cols.copy()
    result_cols.extend(eval_cols)
    result_cols.extend(var_cols)
    result_cols.extend(param_cols)

    # Dataframe for results
    results = pd.DataFrame(columns = result_cols,
                                  index = list(range(max_evals)))
                                  
    #https://stackoverflow.com/questions/20347766/pythonically-add-header-to-a-csv-file
    with open(file_name, 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(result_cols)

    # Keep searching until reach max evaluations
    for i in range(max_evals):
        #picked  hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        hyperparameters.update(kwargs)
        # dictionary for storing average results
        # Store parameter values in results frame
        for k,v in hyperparameters.items():
            results.at[i,k] = v

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
            results.at[i, metric+var_word] = np.std(score_ls)
        
        results.iloc[i, :len(record_cols)] = [i,hyperparameters,np.mean(times)]

        with open(file_name,'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(results.iloc[i,:].values))

        #Print the percentage until finish
        percent_done = roundup(int((i/max_evals)*100))
        if(percent_done in precent_range):
            print(f"Around {percent_done}% done." )
            precent_range.remove(percent_done)

    # Sort with best score on top
    results.sort_values(list(evaluators.keys()), ascending = ascending, inplace = True)
    results.reset_index(inplace = True)

    print("Hyperpameter tuning is done and the best scores are:")
    print(results.loc[0][list(evaluators.keys())]) 
    print("with parameter:", results['params'][0])
    print("Finish tuning in ",round((time.time() - method_start)/60, 2), "minutes.")
    
    return results 



def grid_seach(X, embedder, evaluators:dict, param_grid:dict ,ascending:list,file_name:str,subsampling = False,num_iter:int = 5, random_state = None, **kwargs):
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
    file_name:string
        name of the csv file for saving of the results on disc
    subsampling: optional, bool or int 
        needs to be an int if subsampling is to be used with a certain number of rows, otherwise the full dataset
        will be used. 
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
     #setting up all the columns to be reference later
    var_word = '_variance'
    record_cols = ['iteration', 'params', 'duration_in_second']
    eval_cols = evaluators.keys()
    var_cols = [s+var_word for s in list(evaluators.keys())]
    param_cols = list(param_grid.keys())
    param_cols.extend(list(kwargs.keys()))
    result_cols = record_cols.copy()
    result_cols.extend(eval_cols)
    result_cols.extend(var_cols)
    result_cols.extend(param_cols)

    # Dataframe for results
    results = pd.DataFrame(columns = result_cols,
                                  index = list(range(param_len)))
                                  
    #https://stackoverflow.com/questions/20347766/pythonically-add-header-to-a-csv-file
    with open(file_name, 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(result_cols)
        
    counter = 0

    for i in itertools.product(*values):
        # Retrieving the parameter set for a given value i
        hyperparameters = dict(zip(keys, i))
        hyperparameters.update(kwargs)
        # dictionary for storing average results 
        for k,v in hyperparameters.items():
            results.at[counter,k] = v
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
            metric_results = metrics_scores_iter(sub, embedded, evaluators, return_dict= True,verbose=False)
            for metric, score in metric_results.items():
                scores[metric].append(score)
        for metric, score_ls in scores.items():
            results.at[counter, metric]= np.mean(score_ls)
            results.at[counter, metric+var_word] = np.std(score_ls)
        results.iloc[counter, :len(record_cols)] = [counter,hyperparameters,np.mean(times)]

        with open(file_name,'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(results.iloc[counter,:].values))
    
        #Print the percentage until finish
        percent_done = roundup(int((counter/param_len)*100))
        if(percent_done in precent_range):
            print(f"Around {percent_done}% done." )
            precent_range.remove(percent_done)
        counter+=1

    # Sort with best score on top
    results.sort_values(list(evaluators.keys()), ascending = ascending, inplace = True)
    results.reset_index(inplace = True)

    print("Hyperpameter tuning is done and the best scores are:")
    print(results.loc[0][list(evaluators.keys())])
    print("with parameter:", results['params'][0])
    print("Finish tuning in ",round((time.time() - method_start)/60, 2), "minutes.")
    
    return results 