import sys
sys.path.append("T:/laupodteam/AIOS/Chontira/CellDynClustering")
from ctypes.wintypes import DOUBLE
import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import evaluation.hyperparameter_optimization as hyper
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture




__author__ = "Chontira Chumsaeng"
"""
Unit test to see whether hyperparameter_optimization module works correctly

"""

class Test_hyper_op(unittest.TestCase):
    

    @classmethod
    def setUpClass(cls):
        print("\nSetting up tests for hyperparameter_optimization functions")

    @classmethod
    def tearDownClass(cls):
        print("Finished testing and tearing down the test class\n")

    def setUp(self):

        self.evaluators = {'silhouette_score': silhouette_score,
                            'davies_bouldin_score': davies_bouldin_score}

        self.loose_features,self.loose_clusters = make_blobs(n_samples=5000, n_features = 6, centers = 3, 
                                           cluster_std=0.6, random_state = 0) 
        
        self.best_param = {'n_components': 3,
                            'n_init': 4,
                            'covariance_type': 'spherical',
                            'init_params': 'kmeans'}
                            
        self.param_grid  = {'n_components': [2,3], 
                            'n_init': list(range(2, 10, 2)),
                            'covariance_type' : ('full', 'diag'),
                            'init_params':('k-means++', "kmeans")}

        keys, values = zip(*self.param_grid.items())

        self.param_num_round = 1
        for v in values:
           self.param_num_round*=len(v)

        self.gm = GaussianMixture(**self.best_param)
        
        self.predicted = self.gm.fit_predict(self.loose_features)

        self.unique_label = len(np.unique(self.predicted))

        self.sil = silhouette_score(X = self.loose_features, labels = self.predicted)
        self.db = davies_bouldin_score(X = self.loose_features, labels = self.predicted)
        

    def test_random_search(self):
        ## without undersampling
        print("\nTesting random search")
        results = hyper.random_seach_optimization(self.loose_features, GaussianMixture, self.evaluators, self.param_grid,[False,True],seperating_param_values = True, max_evals=20)
        self.assertEqual(int(results.shape[0]), 20, "Scores are not almost equal")
        self.assertEqual(int(results["num_labels"][0]), self.unique_label, "numbers are not almost equal")

        sil_score= round(self.sil,4)
        sil_best_scores = round(results["silhouette_score"].loc[0], 4)
        sil_best_std_scores = round(results["silhouette_score_std"].loc[0], 4)
        sil_worse_score = round(results["silhouette_score"].loc[19], 4)
        self.assertAlmostEqual(sil_best_scores, sil_score, "Scores are not almost equal")
        
        self.assertTrue(sil_best_scores > sil_worse_score)
        self.assertTrue(sil_best_std_scores == 0)
        
        db_score= round(self.db,4)
        db_best_scores = round(results["davies_bouldin_score"].loc[0], 4)
        db_best_std_scores = round(results["davies_bouldin_score_std"].loc[0], 4)
        db_worse_score = round(results["davies_bouldin_score"].loc[19], 4)
        self.assertAlmostEqual(db_best_scores, db_score, "Scores are not almost equal")
        
        self.assertTrue(db_best_scores < db_worse_score)
        self.assertTrue(db_best_std_scores == 0)
        self.assertFalse(results.isnull().values.any())

        ## testing seperating param values
        for i in range(10):
            param =  results["params"].loc[i]
            for k, v in param.items():
                self.assertEqual(results[k].loc[i], v) 
                
        print("Finish testing random search\n")


    
    def test_grid_search(self):
        ## without undersampling
        print("\nTesting grid search")
        
        results = hyper.grid_seach_optimization(self.loose_features, GaussianMixture, self.evaluators, self.param_grid,[False,True], seperating_param_values = True)
        self.assertEqual(int(results.shape[0]), self.param_num_round, "Scores are not almost equal")
        self.assertEqual(int(results["num_labels"][0]), self.unique_label, "numbers are not almost equal")

        sil_score= round(self.sil,4)
        sil_best_scores = round(results["silhouette_score"].loc[0], 4)
        sil_best_std_scores = round(results["silhouette_score_std"].loc[0], 4)
        sil_worse_score = round(results["silhouette_score"].loc[19], 4)
        self.assertAlmostEqual(sil_best_scores, sil_score, "Scores are not almost equal")
        
        self.assertTrue(sil_best_scores > sil_worse_score)
        self.assertTrue(sil_best_std_scores == 0)
        
        db_score= round(self.db,4)
        db_best_scores = round(results["davies_bouldin_score"].loc[0], 4)
        db_best_std_scores = round(results["davies_bouldin_score_std"].loc[0], 4)
        db_worse_score = round(results["davies_bouldin_score"].loc[19], 4)
        self.assertAlmostEqual(db_best_scores, db_score, "Scores are not almost equal")
        
        self.assertTrue(db_best_scores < db_worse_score)
        self.assertTrue(db_best_std_scores == 0)
        self.assertFalse(results.isnull().values.any())

        ## testing seperating param values
        for i in range(10):
            param =  results["params"].loc[i]
            for k, v in param.items():
                self.assertEqual(results[k].loc[i], v) 

        print("Finish testing grid search\n")

    def tearDown(self):
        del self.evaluators
        del self.param_grid
        del self.loose_clusters
        del self.loose_features
        del self.best_param
        del self.gm
        del self.predicted
        del self.sil
        del self.db
        del self.unique_label

    

if __name__ == "__main__":
    unittest.main()