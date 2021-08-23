"""
Original author: Jille van der Togt
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import umap
import umap.plot
import pacmap
import trimap

from tqdm.notebook import tqdm
from multiprocessing.dummy import Pool as ThreadPool

class Embedding:
	def __init__(self):
		self.embedding_df = None
		self.embedder = None

	def umap_embedding(self, dimensions, n_neighbors=30, min_dist=0.0):
		"""Create embedding
		
		Parameters
		----------
		dataset : pd.DataFrame
			The dataframe to embed
		n_samples : int
			Number of samples to use from dataset
		n_neighbors : int, optional
			Number of neighbors
		min_dist : float, optional
			Minimum distance between points
		dimensions : int, optional
			The number of dimensions to reduce to

		Returns
		-------
		embedding
			embedding fitted on dataset

		"""


		self.embedder = umap.UMAP(
			n_neighbors=n_neighbors,
			min_dist=min_dist,
			n_components=dimensions,
			random_state=42,
		)

	def pca_embedder(self,dimensions):
		self.embedder = PCA()

	def trimap_embedder(self,):
		self.embedder = trimap.TRIMAP()

	def pacmap_embedder(self):
		self.embedder = pacmap.PaCMAP()

	def fit_embedding(self, subset):
		return self.embedder.fit(subset)

	def plot_embedding_2d(self, embedding, labels, features=[0,1], size=0.1):
		"""Plot embedding in 2D
		
		Parameters
		----------
		embedding : list
			A list with n dimenstions
		labels : list, optional
			A list with labels for the datapoints
		features : list, optional
			The indices of the features used to plot
		size : float, optional
			Size of the datapoints in the plot

		Returns
		-------
		plot
			2D plot

		"""
		plt.scatter(embedding[:, features[0]],
					embedding[:, features[1]],
					c=labels,
					s=size,
					cmap='Spectral')
		return plot

	def plot_embedding_3d(self, embedding, labels=None, features=[0,1,2], size=0.1):
		"""Plot embedding in 3D
		
		Parameters
		----------
		embedding : list
			A list with n dimenstions
		labels : list, optional
			A list with labels for the datapoints
		features : list, optional
			The indices of the features used to plot
		size : float, optional
			Size of the datapoints in the plot

		Returns
		-------
		plot
			3D plot

		"""
		fig = plt.figure(1, figsize=(40, 30))
		ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
		ax.scatter(embedding[:, features[0]], embedding[:, features[1]], 
				   embedding[:, features[2]], c=labels, s=size, cmap='tab20')
		ax.tick_params(axis='x', labelsize=30)
		ax.tick_params(axis='y', labelsize=30)
		ax.tick_params(axis='z', labelsize=30)

		return fig
	
	def clean_single(self, embedding, columns, radius, n_neighbors):
		"""Cleans single of noise.
		
		Parameters
		----------
		embedding : list
			A list with n dimenstions
		columns : list
			A list with column headers
		radius : int
			The radius for cleaning
		n_neighbors : int
			Amount of neighbours used to check within radius

		Returns
		-------
		embedding
			Same embedding as input minus noise

		"""
		indices = []
		sampeled = self.embedding_df.sample(n_neighbors*100)
		for index, P in tqdm(enumerate(embedding)):
			distance = np.sqrt(np.square(sampeled-P).sum(axis=1))
			neighbours = np.count_nonzero(distance < radius)
			if neighbours < n_neighbors:
				indices.append(index)
		return indices

	def chunk_it(self, seq, num):
		"""Cuts up a seq into num chunks.

		Parameters
		----------
		seq : list
			List to be cut up into num different sublists
		num : int
			Amount of sublists to be created

		Returns
		-------
		List of num sublists
			Has dimensions: seq dimensios * 2-D list

		"""
		avg = len(seq) / float(num)
		out = []
		last = 0.0

		while last < len(seq):
			out.append(seq[int(last):int(last + avg)])
			last += avg

		return out

	def clean_embedding(self, embedding, n_samples=1000000, n_workers=8, radius=1, n_neighbors=100):
		"""Cleans embedding of noise.
		
		Parameters
		----------
		embedding : list
			A list with n dimenstions
		n_workers : int, optional
			The number of threads the CPU should use
		radius : int, optional
			The radius for cleaning
		n_neighbors : int, optional
			Amount of neighbours used to check within radius

		Returns
		-------
		embedding
			Same embedding as input minus noise

		"""
		columns = ['f' + str(i) for i in range(len(embedding[0]))]

		self.embedding_df = pd.DataFrame(data=embedding,
							index=np.arange(0,len(embedding)),
							columns=columns)

		split_embedding = self.chunk_it(embedding, n_workers)
		pool = ThreadPool(n_workers)
		results = [pool.apply_async(self.clean_single, (subset,columns,radius,n_neighbors)) for subset in split_embedding]
		pool.close()
		pool.join()

		res = [r.get() for r in results]
		for index, i in enumerate(res):
			if index == 0:
				continue
			res[index] = np.array(res[index]) + n_samples/n_workers*index

		res = np.array([item for sublist in res for item in sublist]).astype('int')
		cleaned_embedding = embedding.copy()
		cleaned_embedding = np.delete(cleaned_embedding, res, axis=0)
		return cleaned_embedding, res