__author__ = "Jille Van der Togt" 

import numpy as np
import pandas as pd
from scipy import stats

class load_cell_dyn:
	def __init__(self):
		self.PATH = 'L:/lab_research/RES-Folder-UPOD/Celldynclustering/E_ResearchData/2_ResearchData'
		self.FORMAT = 'sas7bdat'
		self.celldyn_df = None
		self.processed_df = None

	def load_celldyn(self):
		"""Reads celldyn data from sas file.
		
		Parameters
		----------
		-
		Returns
		-------
		-
		"""
		self.celldyn_df = pd.read_sas(self.PATH + '/celldyn.sas7bdat', self.FORMAT)


	def preprocess(self):
		"""Preprocesses data.
		
		Parameters
		----------
		-
		Returns
		-------
		-
		"""
		#Select columns
		self.processed_df = self.celldyn_df.loc[:, [i for i in self.celldyn_df.columns if i[:3] in ('c_b', 'gen', 'age', 'stu')]]
		
		#Drop all rows with 1> NaN (~1m)
		self.processed_df = self.processed_df.dropna()

		#Convert to log-space
		for column in self.processed_df.columns:
			if column[:3] in ('gen', 'age', 'stu'):
				continue
			self.processed_df.loc[:,column] = np.log(self.processed_df[column])

		#Convert gender to integer
		self.processed_df.loc[:, 'gender'] = self.processed_df['gender'].str.decode('UTF-8')
		self.processed_df.loc[:, 'gender'].replace({'M':0, 'F':1, 'O':1}, inplace=True)
		self.processed_df.loc[:, 'gender'] = pd.to_numeric(self.processed_df['gender'])
		self.processed_df.to_csv('data/cell_dyn_processed_frame.csv', index=False)

