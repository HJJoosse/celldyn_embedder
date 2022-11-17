import sys
sys.path.append("T:/laupodteam/AIOS/Chontira/CellDynClustering")
import unittest
from data.load_data import *
import pandas as pd
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


__author__ = "Chontira Chumsaeng"
"""
Unit test to see whether load_data works correctly

"""


class Test_load_data(unittest.TestCase):
    

    @classmethod
    def setUpClass(cls):
        print("Setting up tests for loading data class")

    @classmethod
    def tearDownClass(cls):
        print("Finished testing and loading data the test class")

    def setUp(self):

        self.col_names = ["Feat_1", "Feat_2", "Feat_3", "Feat_4", "Feat_5", "Feat_6"]
        self.col_names_w_labels = ["Feat_1", "Feat_2", "Feat_3", "Feat_4", "Feat_5", "Feat_6", "labels"]
        self.dense_features,self.dense_labels = make_blobs(n_samples=5000, n_features = 6, centers = 3, 
                                           cluster_std=0.6, random_state = 0) 

        self.dense_df = pd.DataFrame(self.dense_features, 
                  columns = self.col_names)

        
                
        self.kmeans = KMeans(n_clusters=4).fit(self.dense_df)
        
        self.dense_predicted = self.kmeans.predict(self.dense_df) 

        self.dense_df.to_csv("data/toy_dense_without_y.csv", index = False)

        self.dense_df.to_feather("data/toy_dense_without_y.feather")
        self.loose_df = pd.read_csv("data/toy_seperated.csv")
        

        self.loose_X = np.asarray(self.loose_df.iloc[:,0:self.loose_df.shape[1]-1])
        self.loose_y = np.asarray(self.loose_df.iloc[:,self.loose_df.shape[1]-1])
        
        self.loaded_data = load_data()
        self.loaded_data.read_from_path("data/toy_seperated.csv", True)

        
    def test_init(self):
        

        """
        check loose data from csv with labels. Fully checking read_from_path
        """
        data_csv = load_data()
        data_csv.read_from_path("data/toy_seperated.csv", True)
        np.testing.assert_array_equal(self.loose_X , data_csv.X)
        np.testing.assert_array_equal(self.loose_y, data_csv.y)
        np.testing.assert_array_equal(np.asarray(data_csv.df), np.asarray(self.loose_df))
        np.testing.assert_array_equal(np.asarray(data_csv.feature_names), np.asarray(self.col_names_w_labels))

        
        """
        check dense data from feather without labels. Fully checking read_from_path
        """
        data_feather_without_labels = load_data()
        data_feather_without_labels.read_from_path("data/toy_dense_without_y.feather", False)
        
        self.assertFalse(data_feather_without_labels.df.empty)
        self.assertTrue(len(data_feather_without_labels.y) == 0)
        self.assertTrue(len(data_feather_without_labels.X) != 0)

        np.testing.assert_array_equal(data_feather_without_labels.X , self.dense_features)
        np.testing.assert_array_equal(np.asarray(data_feather_without_labels.df), np.asarray(self.dense_df))
        np.testing.assert_array_equal(np.asarray(data_feather_without_labels.feature_names), np.asarray(self.col_names))


        #check data chaining 
        data_feather_without_labels.set_y(self.dense_labels).set_predicted(self.dense_predicted)
        
        np.testing.assert_array_equal(data_feather_without_labels.y , self.dense_labels)#
        np.testing.assert_array_equal(data_feather_without_labels.predicted, self.dense_predicted)

        """
        check dense data from csv without labels. Fully checking read_from_path
        """
        data_csv_without_labels = load_data()
        data_csv_without_labels.read_from_path("data/toy_dense_without_y.csv", False)

        self.assertFalse(data_csv_without_labels.df.empty)
        self.assertTrue(len(data_csv_without_labels.y) == 0)
        self.assertTrue(len(data_csv_without_labels.X) != 0)

        self.assertEqual(data_csv_without_labels.X.shape , self.dense_features.shape)
        np.testing.assert_array_equal(np.asarray(data_csv_without_labels.df).shape, np.asarray(self.dense_df).shape)
        np.testing.assert_array_equal(np.asarray(data_csv_without_labels.feature_names), np.asarray(self.col_names))

        """
        check dense data from feather with labels. Only checking for non-empty and empty lists
        """
        data_feather_with_labels = load_data()
        data_feather_with_labels.read_from_path("data/toy_dense_without_y.feather", True)

        self.assertFalse(data_feather_with_labels.df.empty)
        self.assertTrue(len(data_feather_with_labels.y) != 0)
        self.assertTrue(len(data_feather_with_labels.X) != 0)
        np.testing.assert_array_equal(np.asarray(data_feather_with_labels.feature_names), np.asarray(self.col_names))


    def test_set_functions(self):

        new_db = load_data()
        self.assertTrue(len(new_db.X) == 0)
        self.assertTrue(new_db.df.empty)
        self.assertTrue(len(new_db.y) == 0)

        new_db.set_X(self.dense_features)
        np.testing.assert_array_equal(new_db.X , self.dense_features)

        new_db.set_y(self.dense_labels)
        np.testing.assert_array_equal(new_db.y , self.dense_labels)

        new_db.set_predicted(self.dense_predicted)
        np.testing.assert_array_equal(new_db.predicted , self.dense_predicted)

        new_db.set_df(self.dense_df)
        np.testing.assert_array_equal(np.asarray(new_db.df) ,np.asarray(self.dense_df))
        np.testing.assert_array_equal(np.asarray(new_db.feature_names), np.asarray(self.col_names))


    
    def test_set_predicted(self):
        self.loaded_data.set_predicted(np.asarray([23,42,453,54354,3453]))
        np.testing.assert_array_equal(self.loaded_data.predicted, np.asarray([23,42,453,54354,3453]))


    def test_set_df_with_X(self):
        new_db = load_data()
        self.assertTrue(len(new_db.X) == 0)
        self.assertTrue(new_db.df.empty)
        self.assertTrue(len(new_db.y) == 0)

        new_db.set_df_from_X(self.col_names)
        self.assertTrue(new_db.df.empty)

        new_db.set_X(self.dense_features)
        np.testing.assert_array_equal(new_db.X , self.dense_features)

        new_db.set_df_from_X(self.col_names)
        np.testing.assert_array_equal(np.asarray(new_db.df) ,np.asarray(self.dense_df))
        np.testing.assert_array_equal(np.asarray(new_db.feature_names), np.asarray(self.col_names))

        
        
    def test_set_df_with_X_and_y(self):
        new_db = load_data()
        self.assertTrue(len(new_db.X) == 0)
        self.assertTrue(new_db.df.empty)
        self.assertTrue(len(new_db.y) == 0)

        new_db.set_df_from_X_y(self.col_names)
        self.assertTrue(new_db.df.empty)

        new_db.set_X(self.dense_features)
        np.testing.assert_array_equal(new_db.X , self.dense_features)

        new_db.set_y(self.dense_labels)
        np.testing.assert_array_equal(new_db.y , self.dense_labels)

        new_db.set_df_from_X_y(self.col_names)
        self.assertFalse(new_db.df.empty)

        np.testing.assert_array_equal(np.asarray(new_db.feature_names), np.asarray(self.col_names_w_labels))

               

    def test_chaining_of_set_methods(self):
        new_db = load_data()
        self.assertTrue(len(new_db.X) == 0)
        self.assertTrue(new_db.df.empty)
        self.assertTrue(len(new_db.y) == 0)

        new_db.set_X(self.dense_features).set_y(self.dense_labels).set_df_from_X_y(self.col_names)
        np.testing.assert_array_equal(new_db.X , self.dense_features)
        np.testing.assert_array_equal(new_db.y , self.dense_labels)
        
        
    
    def test_chaining_init(self):
        new_db = load_data().read_from_path("data/toy_dense_without_y.feather", False)
        self.assertTrue(len(new_db.y) == 0)

        np.testing.assert_array_equal(new_db.X , self.dense_features)
        np.testing.assert_array_equal(np.asarray(new_db.df) ,np.asarray(self.dense_df))


    def test_chaining_set_df_X(self):

        new_db = load_data()
        self.assertTrue(len(new_db.X) == 0)
        self.assertTrue(new_db.df.empty)
        self.assertTrue(len(new_db.y) == 0)

        new_db.set_X(self.dense_features).set_df_from_X(self.col_names).set_y( self.dense_labels)

        np.testing.assert_array_equal(new_db.X , self.dense_features)
        np.testing.assert_array_equal(np.asarray(new_db.df) ,np.asarray(self.dense_df))
        np.testing.assert_array_equal(new_db.y , self.dense_labels)


    def test_chaining_set_df_X_y(self):

        new_db = load_data()
        self.assertTrue(len(new_db.X) == 0)
        self.assertTrue(new_db.df.empty)
        self.assertTrue(len(new_db.y) == 0)

        new_db.set_X(self.dense_features).set_y(self.dense_labels).set_df_from_X_y(self.col_names).set_predicted(self.dense_predicted)

        np.testing.assert_array_equal(new_db.X , self.dense_features)
        np.testing.assert_array_equal(new_db.y , self.dense_labels)
        np.testing.assert_array_equal(new_db.predicted , self.dense_predicted)

    def test_get_x_y_and_chaing(self):

        new_db = load_data().set_df(self.loaded_data.df).get_x_from_df(True).get_y_from_df()

        np.testing.assert_array_equal(new_db.X , self.loose_X)
        np.testing.assert_array_equal(new_db.y , self.loose_y)

    def test_get_x_and_chaing(self):

        new_db = load_data().set_df(self.loaded_data.df).get_x_from_df(False)
        np.testing.assert_array_equal(new_db.X , np.asarray(self.loose_df))
        self.assertTrue(len(new_db.y) == 0)


    def tearDown(self):
        del self.loaded_data
        del self.dense_df
        del self.dense_features
        del self.dense_labels
        del self.col_names
        del self.kmeans
        del self.dense_predicted
        del self.loose_df
        del self.loose_X 
        del self.loose_y
        del self.col_names_w_labels


    
if __name__ == "__main__":
    unittest.main()
