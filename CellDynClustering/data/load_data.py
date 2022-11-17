import numpy as np
import pandas as pd
import os
import sys
from os.path import exists
import errno

from sklearn.utils.fixes import _object_dtype_isnan
__author__ = "Chontira Chumsaeng"
"""
Class for loading and storing data. All return self so allows chaining of methods.

"""
class load_data(object):

 

    def __init__(self):

        self.df = pd.DataFrame()
        self.y = []
        self.predicted = []
        self.X = []
        self.feature_names = []
    
    
    def read_from_path(self, path, contain_y, **kwargs):

        """
        Read data to pandas then store it as numpy array for X and y
        
        Parameters
        ---------
        path: String
                part to the files

        contain_y: bool
                does the file contain y values
        

        Returns
        ---------
        self: load_data
            for method chaining
        """
        


        if(exists(path)):

            if(path.lower().endswith(".csv")):
                self.df = pd.read_csv(path, **kwargs)
                
            if(path.lower().endswith(".feather")):
                self.df = pd.read_feather(path, **kwargs)

            if(contain_y):
                self.X = np.asarray(self.df)[:,0:self.df.shape[1]-1]
                self.y = np.asarray(self.df)[:,self.df.shape[1]-1]
            else:
                self.X = np.asarray(self.df)
            
            self.feature_names = self.df.columns
        
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

      
        return self

    def set_predicted(self, predicted):
        self.predicted = predicted
        return self


    def set_X(self, X):
        self.X = X
        return self


    def set_y(self, y):
        self.y = y
        return self

    
    def set_df(self, df):
        self.df = df
        self.feature_names = self.df.columns
        return self

    def set_df_from_X(self, column_name):
        if(len(self.X) != 0):
            self.df = pd.DataFrame(self.X, columns=column_name)
            self.feature_names = self.df.columns
        else:
            print("no X to be stored")
        return self


    def set_df_from_X_y(self, column_names):
        if(len(self.X) != 0 and len(self.y) != 0):
            self.df = pd.DataFrame(self.X, columns=column_names)
            self.df["labels"] = self.y
            self.feature_names = self.df.columns
        else:
            print("no X and y to be stored")
        return self

    def get_x_from_df(self, contain_y):

        if(self.df.empty):
            print("no dataframe to extract from")
            return self

        if(contain_y):
            self.X = np.asarray(self.df)[:,0:self.df.shape[1]-1]
        
        else:
            self.X = np.asarray(self.df)
        
        return self

    def get_y_from_df(self):

        if(self.df.empty):
            print("no dataframe to extract from")
        
        else:
            self.y = np.asarray(self.df)[:,self.df.shape[1]-1]
        
        return self

    def save_df(self, path, **kwargs):

        if(self.df.empty):
            print("Dataframe is empty")
        elif(path.lower().endswith(".csv")):
            self.df.to_csv(path=path,**kwargs)
        elif(path.lower().endswith(".feather")):
            self.df.to_feather(path=path,**kwargs)

        return self



    
    
