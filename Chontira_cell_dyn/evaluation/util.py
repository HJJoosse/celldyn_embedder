from tabulate import tabulate
import pandas as pd
import numpy as np
import time
from collections import defaultdict

__author__ = "Chontira Chumsaeng"
"""
Utility functions for evaluating models

"""

def numpy_sampling(X, subsampling):  
    n_data = len(X) 
    idx = np.arange(n_data) 
    np.random.shuffle(idx) 
    return X[idx[: subsampling],:] 

def metrics_scores(x, labels, *evaluator ,verbose=False, return_dict = False):
    """
    Calculates scores for clusters assignment using different metrics (args). 
    
    Parameters
    ---------
    x: dataframe
        data entries for calculating the scores.
    
    labels: list of integers
        cluster labels.
    
    *args: arguments
        further arguments to include the metrics (as function statement)
        if the functions take x and labels as argument.
        
    verbose: bool, optional
        whether to print the results.
    
    
    Returns
    ---------
    results: dict
        dictonary of metrics and their calculated scores.
    
    """
    
    results = {}
    for a in evaluator:
        results.update({a.__name__ : a(x, labels)})
    
    if(verbose):
        print_metric_scores(results)

    
    if(return_dict):
        return results


    
    
def metrics_scores_dict(x, labels, evaluators ,verbose=False, return_dict = False):
    
    """
    Calculates scores for clusters assignment using different metrics (evaluators). 
    
    Parameters
    ---------
    x: dataframe
        data entries for calculating the scores.
    
    labels: list of integers
        cluster labels.
    
    evaluators: dict
        further arguments to include the metrics (as function statement in a dict)
        if the functions take x and labels as argument.
        
    verbose: bool, optional
        whether to print the results.
    
    
    Returns
    ---------
    results: dict
        dictonary of metrics and their calculated scores.
    
    """
    
    results = {}
    for name, metric in evaluators.items():
        results.update({name: metric(x, labels)})
    
    if(verbose):
        print_metric_scores(results)

    
    if(return_dict):
        return results

    

def fit_predict_score(data,model, param,evaluators=None ,verbose=True, return_model_with_label = True, remove_noise = False):

    """
    Feed in a model as a parameter and score the model based on provided metrics 
    (not in unit test)

    Parameter
    --------
    
    data: dataframe
            the data to be fitted to the model.

    model: function (model)
            the model as function to train the dataset.

    evaluators: optional, dict
        further arguments to include the metrics (as function statement in a dict)
        if the functions take x and labels as argument.
    
    params: dict
            the model paramters.

    verbose: bool, optional
        whether to print the results. 
    
    return_model_with_label: bool, optional

    remove_noise: bool, optional
        remove noise from predicted labels
        
    Return
    --------

    labels: list
            the predicted labels from the model.


    """

    start = time.time()

    
    if('metric' in param.keys()):
        if(param['metric'] == 'mahalanobis'):
            V=np.cov(data, rowvar=False)
            model = model(**param, leaf_size = 100, V = V).fit(data)
        else:
            model = model(**param).fit(data)
    else:
        model = model(**param).fit(data) 

    labels = model.fit_predict(data)

    if(remove_noise):
        data = data[labels != -1]
        labels = labels[labels != -1]
    
    if(evaluators!=None):
        metrics_scores_dict(data, labels,evaluators,verbose)

    num_labels = np.unique(labels, return_counts=True)
    print(f"Unique predicted labels {num_labels[0]} with their amounts {num_labels[1]}")
    print(f"Time taken {round((time.time()-start)/60, 2)} minutes")

    if(return_model_with_label):return model, labels
    

    
def subsampling_evaluate_scores(data,model, param,evaluators, subsampling = 20000,num_iters = 10, remove_noise = False):

    """
    Feed in model as a parameter and score the model based on provided metrics 
    using sampling instead of full data and average the scores(not in unit test).

    Parameter
    --------
    
    data: dataframe or numpy array
            the data to be fitted to the model.

    model: function (model)
            the model as function to train the dataset.

     params: dict
            the model paramters.

    evaluators: dict
        further arguments to include the metrics (as function statement in a dict)
        if the functions take x and labels as argument.
    
    subsampling: int
        amount of samples to be subsampled from the full data

    num_iters: int
        number of iterations to run through when scoring the subsampled data

    remove_noise: bool, optional
        remove noise from predicted labels


    """
    data = np.asarray(data)
    method_start = time.time()
    num_labels = []
    scores = defaultdict(list)
    
    for i in range(num_iters):

        sub = numpy_sampling(data, subsampling)
        
        fitted = None
        if('metric' in param.keys()):
            if(param['metric'] == 'mahalanobis'):
                V=np.cov(sub, rowvar=False)
                fitted = model(**param, V = V).fit(sub)
            else:
                fitted = model(**param).fit(sub)
        else:
            fitted = model(**param).fit(sub) 

        labels = fitted.fit_predict(sub)

        if(remove_noise):
            sub = sub[labels != -1]
            labels = sub[labels != -1]
        
        num_labels.append(len(np.unique(labels)))

        metric_results = metrics_scores_dict(sub, labels, evaluators, return_dict= True)
        for metric, score in metric_results.items():
                scores[metric].append(score)
    
    averaged_scores_dict = {}
    for metric, score_ls in scores.items():
            averaged_scores_dict.update({metric:np.mean(score_ls)})
            averaged_scores_dict.update({str(metric) + " standard deviation":np.std(score_ls)})
    
    print_metric_scores(averaged_scores_dict)
    print(f"Unique predicted labels {int(np.mean(num_labels))}")
    print(f"Time taken {round((time.time()-method_start)/60, 2)} minutes")

    



def print_metric_scores(results):
    
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

    