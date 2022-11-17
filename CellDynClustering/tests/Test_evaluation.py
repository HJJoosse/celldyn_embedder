import sys
sys.path.append("T:/laupodteam/AIOS/Chontira/CellDynClustering")
import unittest
from evaluation.util import *
import io
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from tabulate import tabulate


__author__ = "Chontira Chumsaeng"
"""
Unit test to see whether evaluation module works correctly

"""

class Test_evaluation(unittest.TestCase):
    

    @classmethod
    def setUpClass(cls):
        print("Setting up tests for metric evaluation functions")

    @classmethod
    def tearDownClass(cls):
        print("Finished testing and tearing down the test class")

    def setUp(self):

        self.evaluators= {'silhouette_score': silhouette_score,
            'davies_bouldin_score': davies_bouldin_score}

        self.col_names = ["Feat_1", "Feat_2", "Feat_3", "Feat_4", "Feat_5", "Feat_6"]
        self.features,self.labels = make_blobs(n_samples=5000, n_features = 6, centers = 3, 
                                           cluster_std=0.6, random_state = 0) 

        self.df = pd.DataFrame(self.features, 
                  columns = self.col_names)
                
        self.kmeans = KMeans(n_clusters=4).fit(self.df)

        self.param = {'n_clusters' : 4}
        
        self.predicted = self.kmeans.predict(self.df)

        self.sil = silhouette_score(X = self.df, labels = self.predicted)
        self.db = davies_bouldin_score(X = self.df, labels = self.predicted)
        self.results_dict = {"silhouette_score": self.sil, "davies_bouldin_score" : self.db}
        


    def tearDown(self):
        del self.df
        del self.features
        del self.labels
        del self.col_names
        del self.kmeans
        del self.predicted
        del self.sil
        del self.db
        del self.evaluators
        del self.param

    def test_metrics_scores(self):

       
        results = metrics_scores(self.df, self.predicted, silhouette_score, davies_bouldin_score, return_dict=True)
        self.assertEqual(self.sil, results["silhouette_score"], "Silhouette scores are not equal to each other")
        self.assertEqual(self.db, results["davies_bouldin_score"], "Davies Douldin scores are not equal to each other")

        self.assertNotEqual(self.sil, results["davies_bouldin_score"], "Scores should not equal to each other")
        self.assertNotEqual(self.db, results["silhouette_score"], "Scores should not equal to each other")
    

    def test_print_metric_scores(self):

        results = metrics_scores(self.df, self.predicted, silhouette_score, davies_bouldin_score, verbose =False,return_dict=True)
        tab = []

        for met, score in self.results_dict.items():
            tab.append([met, score])

        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput
        print_metric_scores(results)                     #  and redirect stdout.
        sys.stdout = sys.__stdout__  
        get_output = capturedOutput.getvalue()          # Reset redirect.                                       
        #code from https://discuss.dizzycoding.com/python-write-unittest-for-console-print/
        expected = str(tabulate(tab, headers = ["Metric","Score"]))+"\n\n\n"
        print(expected)
        print(get_output)
        self.assertEqual(expected, get_output, "Not the same output")


    def test_metrics_scores_dict(self):

       
        results = metrics_scores_dict(self.df, self.predicted, self.evaluators, return_dict=True)
        self.assertEqual(self.sil, results["silhouette_score"], "Silhouette scores are not equal to each other")
        self.assertEqual(self.db, results["davies_bouldin_score"], "Davies Douldin scores are not equal to each other")

        self.assertNotEqual(self.sil, results["davies_bouldin_score"], "Scores should not equal to each other")
        self.assertNotEqual(self.db, results["silhouette_score"], "Scores should not equal to each other")


    


if __name__ == "__main__":
    unittest.main()