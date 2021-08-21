import os.path
import time
import seaborn as sns
import hdbscan
from collections import Counter
from sklearn import metrics
import pickle
from sklearn import mixture
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn import metrics
from tqdm import tqdm

from embedding import Embedding
from load_data import LoadData
from base import *

def load_data_set():
	"""HELPER FUNCTION: Loads data set
		
		Parameters
		----------
		-

		Returns
		-------
		dataset
			frame with all processed celldyn data

		"""
	loader = LoadData()
	PATH = 'C:/Users/jtogt2/Notebook/data/'

	if os.path.isfile(PATH + 'processed_dataframe.csv'):
		print("Loading saved data...")
		start = time.time()
		loader.load_processed_frame()
		dataset = loader.retrieve_measure_frame()
		print(f"Data loaded in: {time.time() - start} seconds")
	else:
		print("Preprocessing data...")
		start = time.time()
		loader.load_celldyn()
		loader.preprocess()
		dataset = loader.retrieve_measure_frame()
		print(f"Data processed in: {time.time() - start} seconds")

	return dataset

def load_embedding(subset_wo_ga, dimensions, n_neighbors, min_dist):
	"""HELPER FUNCTION: loads embedding
		
		Parameters
		----------
		subset_wo_ga : dataframe
			Processed celldyn data without age and gender columns
		dimensions : int
			The number of dimensions used in training/loading the embedding
		n_neighbors : int
			The number of neighbors used in training/loading the embedding
		min_dist : float
			The minimum distance used in training/loading the embedding

		Returns
		-------
		embedding
			Embedding trained on full processed celldyn data

		"""
	PATH = 'C:/Users/jtogt2/Notebook/data/embeddings/'
	embedder = Embedding()
	
	if os.path.isfile(f'{PATH}embedding_{dimensions}_{n_neighbors}'):
		print("Loading saved embedding...")
		start = time.time()
		embedding = pickle.load((open(f'{PATH}embedding_{dimensions}_{n_neighbors}', 'rb')))
		print(f"Embedding loaded in: {time.time() - start} seconds")
	
	else:
		print("Training embedding...")
		start = time.time()
		untrained_embedding = embedder.create_embedding(dimensions, n_neighbors, min_dist)
		embedding = embedder.fit_embedding(untrained_embedding, subset_wo_ga)
		print(f"Embeddings trained in: {time.time() - start} seconds")
		
		print("Saving embedding...")
		start = time.time()
		pickle.dump(embedding, open(f'{PATH}embedding_{dimensions}_{n_neighbors}', 'wb'))
		print(f"Embedding saved in: {time.time() - start} seconds")

	return embedding

def load_labels(embedded_dataset, dimensions, min_samples, min_cluster_size):
	"""HELPER FUNCTION: loads data labels
		
		Parameters
		----------
		embedded_dataset : list
			A list with the full data set ran trough an embedding
		dimensions : int
			The number of dimensions used to create/load the cluster-labels
		min_samples : int
			The minimum amount of samples used to create/load the cluster-labels
		min_clsuter_size : int
			The minimum cluster size used to create/load the cluster labels

		Returns
		-------
		labels : list
			List with all cluster labels for all samples

		"""
	PATH = 'C:/Users/jtogt2/Notebook/data/embeddings/'
	cleaned_df = pd.DataFrame(data=embedded_dataset,
						  index=np.arange(0,len(embedded_dataset)),
						  columns=['f' + str(i) for i in range(len(embedded_dataset[0]))])

	if os.path.isfile(f'{PATH}embedding_{dimensions}_{min_samples}_{min_cluster_size}_labels.csv'):
		labels = np.genfromtxt(f'{PATH}embedding_{dimensions}_{min_samples}_{min_cluster_size}_labels.csv', delimiter=',')
	else:
		print("Clustering embedding...")
		start = time.time()
		labels = hdbscan.HDBSCAN(
			min_samples=min_samples,
			min_cluster_size=min_cluster_size,
		).fit_predict(cleaned_df)
		print(f"Embedding clustered in: {time.time() - start} seconds")
							   
		print("Saving embedding cluster labels...")
		start = time.time()
		np.savetxt(f'{PATH}embedding_{dimensions}_{min_samples}_{min_cluster_size}_labels.csv', labels, delimiter=",")
		print(f"Embedding cluster labels saved in: {time.time() - start} seconds")
	
	return labels

def calculate_scores(subset, labels):
	"""HELPER FUNCTION: calculates scores for metrics over embedded clusters
		
		Parameters
		----------
		subset : dataframe
			The dataframe with all entries
		labels : list
			List of all cluster-labels corresponding to entries in 'subset'

		Returns
		-------
		scores : list
			List with silhouette score, davies-bouldin index, and dunn index

		"""
	start = time.time()
	silhouette_scores = []
	for _ in tqdm(range(10)):
		silhouette_scores.append(metrics.silhouette_score(subset, labels, sample_size=30000))
	
	davies_scores = metrics.davies_bouldin_score(subset, labels)
	
	dunn_scores = []
	for _ in tqdm(range(5)):
		sub = subset.sample(35000)
		ind = sub.index
		sub_labels = labels.take(ind)
		dunn_scores.append(dunn_fast(sub, sub_labels))
	print(f"Scores calculated in {time.time()-start} seconds")

	return [np.mean(silhouette_scores), davies_scores, np.mean(dunn_scores)]

def calculate_plots(cleaned_dataset):
	"""HELPER FUNCTION: calculates boxplots for all clusters
		
		Parameters
		----------
		cleaned_dataset : dataframe
			Dataframe with all entries and corresponding cluster-labels

		Returns
		-------
		-

		"""
	PATH = 'C:/Users/jtogt2/Notebook/cluster_analysis/plots/'
	grouped_cleaned = np.array(cleaned_dataset.groupby('cluster_assignment'))
	for cluster in grouped_cleaned:
		if cluster[1].shape[0] > 80000:
			fig = px.box(cluster[1].sample(80000))
		else: 
			fig = px.box(cluster[1])
		fig.write_html(PATH + f"cluster_{cluster[0]}.html",
					  full_html=False,
					  include_plotlyjs='cdn')

def calculate_frames(cleaned_dataset, labels, subset_wi_ga, dimensions, save=False):
	"""HELPER FUNCTION: calculates mean blood views for all clusters
		
		Parameters
		----------
		cleaned_dataset : dataframe
			Dataframe with all entries and corresponding cluster-labels
		labels : list
			List with all labels for all data entries
		subset_wi_ga : dataframe
			Dataframe with all data including age and gender columns
		dimensions : int
			Number of dimensions used for saving blood views under correct name
		save : bool, optional
			Used to either save the frame or not


		Returns
		-------
		result : dataframe
			Dataframe with blood view for all clusters

		"""
	li = Counter(labels)
	li = np.array(sorted(li.items()))
	if li[0][0] == -1:
		li = li[1:,1]
	else:
		li = li[:,1]

	cleaned_dataset['gender'] = subset_wi_ga['gender']
	cleaned_dataset['age'] = subset_wi_ga['age']
	cleaned_dataset['ID'] = subset_wi_ga['studyId_Alle_celldyn']

	unclassfied = cleaned_dataset[cleaned_dataset['cluster_assignment'] == -1].shape[0]
	print(f"HDBSCAN does not classify {unclassfied} samples")
	cleaned_dataset = cleaned_dataset[cleaned_dataset['cluster_assignment'] > -1]

	clustered = cleaned_dataset.groupby('cluster_assignment')
	id_counts = []
	for i in clustered:
	    if i[0] == -1:
	        continue
	    else:
	        ids_counter = len(Counter(i[1]['ID']))
	        id_counts.append(ids_counter)

	PATH = 'C:/Users/jtogt2/Notebook/cluster_analysis/frames/'
	mean = cleaned_dataset.groupby('cluster_assignment').mean()
	mean.insert(0, 'cluster size', li)
	mean['ID'] = id_counts
	mean = mean.rename(columns={'ID':'n_people'})

	result = mean.sort_values(by=['cluster size'], ascending=False)

	if save:
		result.to_excel(f'{PATH}result_cluster_{dimensions}.xlsx')

	return result

def run_dimension(dimensions, n_neighbors=30, min_dist=0.0, min_samples=30, min_cluster_size=1000, plots=False, scores=False, frames=False, save=False):
	"""Runs all helper functions and does everything
		
		Parameters
		----------
		dimensions : int
			Number of dimensions used in creating the embedding, also used for saving purposes
		n_neighbors : int, optional
			Number of neighbors used in creating the embedding, also used for saving purposes
		min_dist : float, optional
			Minimum distance used in creating the embedding, also used for saving purposes
		min_samples : int, optional
			Minimum samples used in creating the clustering, also used for saving purposes
		min_cluster_size : int, optional
			Minimum cluster size used in creating the clustering, also used for saving purposes
		plots : bool, optional
			To create the boxplots for all clusters
		scores : bool, optional
			To create the score metrics for the full clustered embedding
		frames : bool, optional
			To create the average blood view for all clusters
		save : bool, optional
			To save the average blood view for all clusters in an excel sheet

		Returns
		-------
		frames : dataframe
			Dataframe with average blood view for all clusters
		scores : list
			List with silhouette score, davies-bouldin index, and dunn index for current embedding

		"""
	subset_wi_ga = load_data_set()
	subset_wo_ga = subset_wi_ga.loc[:, [i for i in subset_wi_ga.columns if i[:3] == 'c_b']]
	embedding = load_embedding(subset_wo_ga, dimensions, n_neighbors, min_dist)
	embedded_dataset = embedding.transform(subset_wo_ga)
	labels = load_labels(embedded_dataset, dimensions, min_samples, min_cluster_size)
	subset = subset_wo_ga.reset_index(drop=True)

	cleaned_dataset = np.e**subset
	cleaned_dataset['cluster_assignment'] = labels

	if frames:
		return calculate_frames(cleaned_dataset, labels, subset_wi_ga, dimensions, save)
	if plots:
		calculate_plots(cleaned_dataset)
	if scores:
		return calculate_scores(subset, labels)