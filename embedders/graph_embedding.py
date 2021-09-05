from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.neighbors import NearestNeighbors as KNN  # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

import cvxpy as cp
from scipy.sparse.csgraph import laplacian
import scipy as sc
from sklearn.neighbors import NearestNeighbors
import math

from tqdm import tqdm

'''
[x] Sammon
[x] MaximumVarianceUnfolding
[x] LandmarkMDS

[] Node2Vec, pytorch geometric
[] Graph2Vec, https://karateclub.readthedocs.io/en/latest/_modules/karateclub/graph_embedding/graph2vec.html
[] Role2Vec, https://karateclub.readthedocs.io/en/latest/_modules/karateclub/node_embedding/structural/role2vec.html
[] DeepWalk, https://karateclub.readthedocs.io/en/latest/_modules/karateclub/node_embedding/neighbourhood/deepwalk.html
[] BoostNE, https://karateclub.readthedocs.io/en/latest/_modules/karateclub/node_embedding/neighbourhood/boostne.html
[] Diff2Vec, https://karateclub.readthedocs.io/en/latest/_modules/karateclub/node_embedding/neighbourhood/diff2vec.html
[] LINE, https://github.com/DMPierre/LINE, https://karateclub.readthedocs.io/en/latest/_modules/karateclub/graph_embedding/gl2vec.html
[] SDNE
[] Sub2Vec
[] AttentionWalk, https://github.com/benedekrozemberczki/AttentionWalk/tree/master/src


[] MCML : https://github.com/pachterlab/MCML
[] metapath2vec, pytorch geometric
[] ARGA, pytorch geometric
[] GAE, pytorch geometric
[] SageConv, pytorch geometric
[] PyTorch BigGraph, https://medium.com/dataseries/facebooks-pygraph-is-an-open-source-framework-for-capturing-knowledge-in-large-graphs-b52c0fb902e8
'''

class DisconnectError(Exception):
    """
    An error class to catch if the graph has unconnected regions.
    """

    def __init__(self, message):
        self.message = message


import warnings
warnings.filterwarnings("ignore")

class Sammon(BaseEstimator, TransformerMixin):
    # source: https://github.com/bghojogh/MDS-SammonMapping-Isomap

    def __init__(self, n_components, n_neighbors=None, 
                 max_iterations=100, learning_rate=0.1, 
                 min_score=0.00025, decay_rate=0.75, 
                 init_type="PCA"):
        self.embedding_dimensionality = n_components
        self.n_neighbors = n_neighbors
        self.n_neighbors = n_neighbors
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.min_score = min_score
        self.decay_rate = decay_rate
        self.init_type = init_type  #--> PCA, random

    def fit(self, X, y=None):
         # X: samples are put column-wise in matrix
        X = np.transpose(X)
        self.n_dimensions = X.shape[0]
        self.n_samples = X.shape[1]
        if self.n_neighbors is None:
            self.n_neighbors = self.n_samples - 1 
        self.X_transformed=self.Quasi_Newton_optimization(X=X, max_iterations=self.max_iterations)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return np.transpose(self.X_transformed)

    def Quasi_Newton_optimization(self, X, max_iterations=100):
        # X: column-wise samples
        iteration_start = 0
        scores = []
        delta_score = 1
        if self.init_type == "random":
            X_low_dim = np.random.rand(self.embedding_dimensionality, self.n_samples)  # --> rand in [0,1)
        elif self.init_type == "PCA":
            pca = PCA(n_components=self.embedding_dimensionality)
            X_low_dim = (pca.fit_transform(X.T)).T
        KNN_distance_matrix_initial, neighbors_indices = self.find_KNN_distance_matrix(X=X, n_neighbors=self.n_neighbors)
        normalization_factor = sum(sum(KNN_distance_matrix_initial))
        for iteration_index in range(iteration_start, max_iterations):
            print("Performing quasi Newton, iteration " + str(iteration_index))
            All_NN_distance_matrix, _ = self.find_KNN_distance_matrix(X=X_low_dim, n_neighbors=self.n_samples-1)
            for sample_index in range(self.n_samples):
                for dimension_index in range(self.embedding_dimensionality):
                    # --- calculate gradient and second derivative of gradient (Hessian):
                    gradient_term = 0.0
                    Hessian_term = 0.0
                    for neighbor_index in range(self.n_neighbors):
                        neighbor_index_in_dataset = neighbors_indices[sample_index, neighbor_index]
                        d = All_NN_distance_matrix[sample_index, neighbor_index_in_dataset]
                        d_initial = KNN_distance_matrix_initial[sample_index, neighbor_index_in_dataset]
                        gradient_term += ((d - d_initial) / (d * d_initial)) * (X_low_dim[dimension_index, sample_index] - X_low_dim[dimension_index, neighbor_index_in_dataset])
                        Hessian_term += ((d - d_initial) / (d * d_initial)) - ((X_low_dim[dimension_index, sample_index] - X_low_dim[dimension_index, neighbor_index_in_dataset])**2 / d**3)
                    gradient_term *= (1 / normalization_factor)
                    Hessian_term *= (1 / normalization_factor)
                    gradient_ = gradient_term
                    Hessian_ = Hessian_term
                    # --- update solution:
                    X_low_dim[dimension_index, sample_index] = X_low_dim[dimension_index, sample_index] - (self.learning_rate * abs(1/Hessian_) * gradient_)
            # calculate the objective function:
            objective_function_distance_part = 0.0
            for sample_index in range(self.n_samples):
                temp_ = 0.0
                for neighbor_index in range(self.n_neighbors):
                    neighbor_index_in_dataset = neighbors_indices[sample_index, neighbor_index]
                    d = All_NN_distance_matrix[sample_index, neighbor_index_in_dataset]
                    d_initial = KNN_distance_matrix_initial[sample_index, neighbor_index_in_dataset]
                    temp_ += (d - d_initial)**2 / d_initial
                objective_function_distance_part += (1 / normalization_factor) * temp_
            objective_function = 0.5 * objective_function_distance_part

            if (objective_function<self.min_score):
                return X_low_dim
            elif (delta_score)<0.5 & (delta_score>0):
                idecay_rate = 1/self.decay_rate
                print(f"Relative delta score dropped below 0.5%, \
                        increasing learning rate to {str(idecay_rate)}")
                self.learning_rate = self.learning_rate*(idecay_rate)
            elif (delta_score<0):
                return X_low_dim
                #self.learning_rate = self.learning_rate*self.decay_rate

            scores.append(objective_function) 
            if len(scores)>1:                         
                delta_score = 100*(scores[-2]-scores[-1])/scores[-2] 
                print("iteration " + str(iteration_index) + ": objective cost\
                                = " + str(objective_function), ": decrease = "+str(delta_score))

        return X_low_dim

    def find_KNN_distance_matrix(self, X, n_neighbors):
        # X: column-wise samples
        # returns KNN_distance_matrix: row-wise --> shape: (n_samples, n_samples) where zero for not neighbors
        # returns neighbors_indices: row-wise --> shape: (n_samples, n_neighbors)
        knn = KNN(n_neighbors=n_neighbors+1, algorithm='kd_tree', n_jobs=-1)  #+1 because the point itself is also counted
        knn.fit(X=X.T)
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors_graph
        # the following function gives n_samples*n_samples matrix, and puts 0 for diagonal and also where points are not connected directly in KNN graph
        # if K=n_samples, only diagonal is zero.
        Euclidean_distance_matrix = knn.kneighbors_graph(X=X.T, n_neighbors=n_neighbors+1, mode='distance') #--> gives Euclidean distances
        KNN_distance_matrix = Euclidean_distance_matrix.toarray()
        neighbors_indices = np.zeros((KNN_distance_matrix.shape[0], n_neighbors))
        for sample_index in range(KNN_distance_matrix.shape[0]):
            neighbors_indices[sample_index, :] = np.ravel(np.asarray(np.where(KNN_distance_matrix[sample_index, :] != 0)))
        neighbors_indices = neighbors_indices.astype(int)
        return KNN_distance_matrix, neighbors_indices

    def remove_outliers(self, data_, color_meshgrid):
        # data_: column-wise samples
        data_outliers_removed = data_.copy()
        color_meshgrid_outliers_removed = color_meshgrid.copy()
        for dimension_index in range(data_.shape[0]):
            data_dimension = data_[dimension_index, :].ravel()
            # Set upper and lower limit to 3 standard deviation
            data_dimension_std = np.std(data_dimension)
            data_dimension_mean = np.mean(data_dimension)
            anomaly_cut_off = data_dimension_std * 3
            lower_limit = data_dimension_mean - anomaly_cut_off
            upper_limit = data_dimension_mean + anomaly_cut_off
            samples_to_keep = []
            for sample_index in range(data_outliers_removed.shape[1]):
                sample_ = data_outliers_removed[:, sample_index]
                if sample_[dimension_index] > upper_limit or sample_[dimension_index] < lower_limit:
                    samples_to_keep.append(False)
                else:
                    samples_to_keep.append(True)
            data_outliers_removed = data_outliers_removed.compress(samples_to_keep, axis=1)
            color_meshgrid_outliers_removed = color_meshgrid_outliers_removed.compress(samples_to_keep)
        return data_outliers_removed, color_meshgrid_outliers_removed

    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()

    def load_variable(self, name_of_variable, path='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable

    def save_np_array_to_txt(self, variable, name_of_variable, path_to_save='./'):
        if type(variable) is list:
            variable = np.asarray(variable)
        # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open(file_address, 'w') as f:
            f.write(np.array2string(variable, separator=', '))
            

class MaximumVarianceUnfolding(BaseEstimator, TransformerMixin):
    # source: https://github.com/calebralphs/maximum-variance-unfolding

    def __init__(self, equation="berkley", solver=cp.SCS, solver_tol=1e-2,
                 eig_tol=1.0e-10, solver_iters=2500, dropout_rate=.2, 
                 n_neighbors=7, n_components=2, warm_start=False, seed=None):
        """
        :param equation: A string either "berkley" or "wikipedia" to represent
                         two different equations for the same problem.
        :param solver: A CVXPY solver object.
        :param solver_tol: A float representing the tolerance the solver uses to know when to stop.
        :param eig_tol: The positive semi-definite constraint is only so accurate, this sets
                        eigenvalues that lie in -eig_tol < 0 < eig_tol to 0.
        :param solver_iters: The max number of iterations the solver will go through.
        :param warm_start: Whether or not to use a warm start for the solver.
                           Useful if you are running multiple tests on the same data.
        :param seed: The numpy seed for random numbers.
        """
        self.equation = equation
        self.solver = solver
        self.solver_tol = solver_tol
        self.eig_tol = eig_tol
        self.solver_iters = solver_iters
        self.warm_start = warm_start
        self.seed = seed
        self.neighborhood_graph = None
        self.dropout_rate = dropout_rate
        self.n_neighbors = n_neighbors
        self.n_components = n_components

    def fit(self, data, y=None):
        """
        The method to fit an MVU model to the data.
        :param data: The data to which the model will be fitted.
        :param k: The number of neighbors to fix.
        :param dropout_rate: The number of neighbors to discount.
        :return: Embedded Gramian: The Gramian matrix of the embedded data.
        """
        # Number of data points in the set
        n = data.shape[0]

        # Set the seed
        np.random.seed(self.seed)

        # Calculate the nearest neighbors of each data point and build a graph
        N = NearestNeighbors(n_neighbors=self.n_neighbors).fit(data)\
                                                          .kneighbors_graph(data)\
                                                          .todense()
        N = np.array(N)

        # Randomly drop certain connections.
        # Not the most efficient way but with this implementation random
        #  cuts that disconnect the graph will be caught.
        for i in range(n):
            for j in range(n):
                if N[i, j] == 1 and np.random.random() < self.dropout_rate:
                    N[i, j] = 0.

        # Save the neighborhood graph to be accessed latter
        self.neighborhood_graph = N

        # To check for disconnected regions in the neighbor graph
        lap = laplacian(N, normed=True)
        eigvals, _ = np.linalg.eig(lap)

        for e in eigvals:
            if e == 0. and self.solver_iters is None:
                raise DisconnectError("DISCONNECTED REGIONS IN NEIGHBORHOOD GRAPH. "
                                      "PLEASE SPECIFY MAX ITERATIONS FOR THE SOLVER")

        # Declare some CVXPy variables
        # Gramian of the original data
        P = cp.Constant(data.dot(data.T))
        # The projection of the Gramian
        Q = cp.Variable((n, n), PSD=True)
        # Initialized to zeros
        Q.value = np.zeros((n, n))
        # A shorter way to call a vector of 1's
        ONES = cp.Constant(np.ones((n, 1)))
        # A variable to keep the notation consistent with the Berkley lecture
        T = cp.Constant(n)

        # Declare placeholders to get rid of annoying warnings
        objective = None
        constraints = []

        # Wikipedia Solution
        if self.equation == "wikipedia":
            objective = cp.Maximize(cp.trace(Q))

            constraints = [Q >> 0, cp.sum(Q, axis=1) == 0]

            for i in range(n):
                for j in range(n):
                    if N[i, j] == 1:
                        constraints.append((P[i, i] + P[j, j] - P[i, j] - P[j, i]) -
                                           (Q[i, i] + Q[j, j] - Q[i, j] - Q[j, i]) == 0)

        # UC Berkley Solution
        if self.equation == "berkley":
            objective = cp.Maximize(cp.multiply((1 / T), cp.trace(Q)) -
                                    cp.multiply((1 / (T * T)), cp.trace(cp.matmul(cp.matmul(Q, ONES), ONES.T))))

            constraints = [Q >> 0, cp.sum(Q, axis=1) == 0]
            for i in range(n):
                for j in range(n):
                    if N[i, j] == 1.:
                        constraints.append(Q[i, i] - 2 * Q[i, j] + Q[j, j] -
                                           (P[i, i] - 2 * P[i, j] + P[j, j]) == 0)

        # Solve the problem with the SCS Solver
        problem = cp.Problem(objective, constraints)
        # FIXME The solvertol syntax is unique to SCS
        problem.solve(solver=self.solver,
                      eps=self.solver_tol,
                      max_iters=self.solver_iters,
                      warm_start=self.warm_start)

        return Q.value

    def fit_transform(self, data, y=None):
        """
        The method to fit and transform an MVU model to the data.
        :param data: The data to which the model will be fitted.
        :param dim: The new dimension of the dataset.
        :param k: The number of neighbors to fix.
        :param dropout_rate: The number of neighbors to discount.
        :return: embedded_data: The embedded form of the data.
        """

        embedded_gramian = self.fit(data)

        # Retrieve Q
        embedded_gramian = embedded_gramian

        # Decompose gramian to recover the projection
        eigenvalues, eigenvectors = np.linalg.eig(embedded_gramian)

        # Set the eigenvalues that are within +/- eig_tol to 0
        eigenvalues[np.logical_and(-self.eig_tol < eigenvalues, eigenvalues < self.eig_tol)] = 0.

        # Assuming the eigenvalues and eigenvectors aren't sorted,
        #    sort them and get the top "dim" ones
        sorted_indices = eigenvalues.argsort()[::-1]
        top_eigenvalue_indices = sorted_indices[:self.n_components]

        # Take the top eigenvalues and eigenvectors
        top_eigenvalues = eigenvalues[top_eigenvalue_indices]
        top_eigenvectors = eigenvectors[:, top_eigenvalue_indices]

        # Some quick math to get the projection and return it
        lbda = np.diag(top_eigenvalues ** 0.5)
        embedded_data = lbda.dot(top_eigenvectors.T).T

        return embedded_data


class LandmarkMaximumVarianceUnfolding(BaseEstimator, TransformerMixin):
    # source: https://github.com/calebralphs/maximum-variance-unfolding

    def __init__(self, equation="berkley", n_landmarks=50, solver=cp.SCS, solver_tol=1e-2,
                 eig_tol=1.0e-10, solver_iters=2500, n_neighbors=7, n_components=2,
                 warm_start=False, seed=None):
        """
        :param equation: A string either "berkley" or "wikipedia" to represent
                         two different equations for the same problem.
        :param landmark: None if you do not want to use landmark MVU, otherwise the number of landmarks to consider.
        :param solver: A CVXPY solver object.
        :param solver_tol: A float representing the tolerance the solver uses to know when to stop.
        :param eig_tol: The positive semi-definite constraint is only so accurate, this sets
                        eigenvalues that lie in -eig_tol < 0 < eig_tol to 0.
        :param solver_iters: The max number of iterations the solver will go through.
        :param warm_start: Whether or not to use a warm start for the solver.
                           Useful if you are running multiple tests on the same data.
        :param seed: The numpy seed for random numbers.
        """
        self.equation = equation
        self.n_landmarks = n_landmarks
        self.solver = solver
        self.solver_tol = solver_tol
        self.eig_tol = eig_tol
        self.solver_iters = solver_iters
        self.warm_start = warm_start
        self.seed = seed
        self.neighborhood_graph = None
        self.n_neighbors = n_neighbors
        self.n_components = n_components

    def fit(self, data, y=None):
        """
        The method to fit an MVU model to the data.
        :param data: The data to which the model will be fitted.
        :param k: The number of neighbors to fix.
        :return: Embedded Gramian: The Gramian matrix of the embedded data.
        """
        # Number of data points in the set
        n = data.shape[0]

        # Set the seed
        np.random.seed(self.seed)

        # Calculate the nearest neighbors of each data point and build a graph
        N = NearestNeighbors(n_neighbors=self.n_neighbors).fit(data)\
                                                    .kneighbors_graph(data)\
                                                    .todense()
        N = np.array(N)

        # Sort the neighbor graph to find the points with the most connections
        num_connections = N.sum(axis=0).argsort()[::-1]

        # Separate the most popular points
        top_landmarks_idxs = num_connections[:self.n_landmarks]
        top_landmarks = data[top_landmarks_idxs, :]

        # Compute the nearest neighbors for all of the landmarks so they are all connected
        L = NearestNeighbors(n_neighbors=3).fit(top_landmarks).kneighbors_graph(top_landmarks).todense()
        L = np.array(L)

        # The data without the landmarks
        new_data_idxs = [x for x in list(range(n)) if x not in top_landmarks_idxs]
        new_data = np.delete(data, top_landmarks_idxs, axis=0)

        # Construct a neighborhood graph where each point finds its closest landmark
        l = NearestNeighbors(n_neighbors=3).fit(top_landmarks)\
                                            .kneighbors_graph(new_data)\
                                            .todense()
        l = np.array(l)
        print("shape l", l.shape)

        # Reset N to all 0's
        N = np.zeros((n, n))

        # Add all of the intra-landmark connections to the neighborhood graph
        for i in range(self.n_landmarks):
            for j in range(self.n_landmarks):
                if L[i, j] == 1.:
                    N[top_landmarks_idxs[i], top_landmarks_idxs[j]] = 1.

        # Add all of the inter-landmark connections to the neighborhood graph
        for i in range(n - self.n_landmarks):
            for j in range(self.n_landmarks):
                if l[i, j] == 1.:
                    N[new_data_idxs[i], top_landmarks_idxs[j]] = 1.

        # Save the neighborhood graph to be accessed latter
        self.neighborhood_graph = N

        # To check for disconnected regions in the neighbor graph
        lap = laplacian(N, normed=True)
        eigvals, _ = np.linalg.eig(lap)

        for e in eigvals:
            if e == 0. and self.solver_iters is None:
                raise DisconnectError("DISCONNECTED REGIONS IN NEIGHBORHOOD GRAPH. "
                                      "PLEASE SPECIFY MAX ITERATIONS FOR THE SOLVER")

        # Declare some CVXPy variables
        # Gramian of the original data
        P = cp.Constant(data.dot(data.T))
        # The projection of the Gramian
        Q = cp.Variable((n, n), PSD=True)
        # Initialized to zeros
        Q.value = np.zeros((n, n))
        # A shorter way to call a vector of 1's
        ONES = cp.Constant(np.ones((n, 1)))
        # A variable to keep the notation consistent with the Berkley lecture
        T = cp.Constant(n)

        # Declare placeholders to get rid of annoying warnings
        objective = None
        constraints = []

        # Wikipedia Solution
        if self.equation == "wikipedia":
            objective = cp.Maximize(cp.trace(Q))

            constraints = [Q >> 0, cp.sum(Q, axis=1) == 0]

            for i in range(n):
                for j in range(n):
                    if N[i, j] == 1:
                        constraints.append((P[i, i] + P[j, j] - P[i, j] - P[j, i]) -
                                           (Q[i, i] + Q[j, j] - Q[i, j] - Q[j, i]) == 0)

        # UC Berkley Solution
        if self.equation == "berkley":
            objective = cp.Maximize(cp.multiply((1 / T), cp.trace(Q)) -
                                    cp.multiply((1 / (T * T)), cp.trace(cp.matmul(cp.matmul(Q, ONES), ONES.T))))

            constraints = [Q >> 0, cp.sum(Q, axis=1) == 0]
            for i in range(n):
                for j in range(n):
                    if N[i, j] == 1.:
                        constraints.append(Q[i, i] - 2 * Q[i, j] + Q[j, j] -
                                           (P[i, i] - 2 * P[i, j] + P[j, j]) == 0)

        # Solve the problem with the SCS Solver
        problem = cp.Problem(objective, constraints)
        # FIXME The solvertol syntax is unique to SCS
        problem.solve(solver=self.solver,
                      eps=self.solver_tol,
                      max_iters=self.solver_iters,
                      warm_start=self.warm_start)

        self.embedded_gramian = Q.value
        return self

    def fit_transform(self, data, y=None):
        """
        The method to fit and transform an MVU model to the data.
        :param data: The data to which the model will be fitted.
        :param dim: The new dimension of the dataset.
        :param k: The number of neighbors to fix.
        :return: embedded_data: The embedded form of the data.
        """

        self.fit(data)
        # Decompose gramian to recover the projection
        eigenvalues, eigenvectors = np.linalg.eig(self.embedded_gramian)

        # Set the eigenvalues that are within +/- eig_tol to 0
        eigenvalues[np.logical_and(-self.eig_tol
                     < eigenvalues, eigenvalues < self.eig_tol)] = 0.

        # Assuming the eigenvalues and eigenvectors aren't sorted,
        #    sort them and get the top "dim" ones
        sorted_indices = eigenvalues.argsort()[::-1]
        top_eigenvalue_indices = sorted_indices[:self.n_components]

        # Take the top eigenvalues and eigenvectors
        top_eigenvalues = eigenvalues[top_eigenvalue_indices]
        top_eigenvectors = eigenvectors[:, top_eigenvalue_indices]

        # Some quick math to get the projection and return it
        lbda = np.diag(top_eigenvalues ** 0.5)
        embedded_data = lbda.dot(top_eigenvectors.T).T

        return embedded_data


class LandmarkMultiDimensionalScaling(BaseEstimator, TransformerMixin):
    #  source: https://github.com/danilomotta/LMDS

    def __init__(self,  n_components=2, n_landmarks=100):
        self.n_components = n_components
        self.n_landmarks = n_landmarks

    def fit(self, X, y=0):
        self.landmarks = np.random.randint(0, X.shape[0], self.n_landmarks)
        self.D = sc.spatial.distance.cdist(X[self.landmarks,:], X, 'euclidean')        

        self.Dl = self.D[:,self.landmarks]
        n = len(self.Dl)

        # Centering matrix
        H = - np.ones((n, n))/n
        np.fill_diagonal(H,1-1/n)
        # YY^T
        H = -H.dot(self.Dl**2).dot(H)/2

        # Diagonalize
        evals, evecs = np.linalg.eigh(H)

        # Sort by eigenvalue in descending order
        idx   = np.argsort(evals)[::-1]
        self.evals = evals[idx]
        self.evecs = evecs[:,idx]       
        return self

    def fit_transform(self, X, y=0):
        self.fit(X)  
        # Compute the coordinates using positive-eigenvalued components only
        w, = np.where(self.evals > 0)
        if self.n_components:
            arr = self.evals
            w = arr.argsort()[-self.n_components:][::-1]
            if np.any(self.evals[w]<0):
                print('Error: Not enough positive eigenvalues for the selected dim.')
                return []
        if w.size==0:
            print('Error: matrix is negative definite.')
            return []

        V = self.evecs[:,w]
        L = V.dot(np.diag(np.sqrt(self.evals[w]))).T
        N = self.D.shape[1]
        Lh = V.dot(np.diag(1./np.sqrt(self.evals[w]))).T
        Dm = self.D - np.tile(np.mean(self.Dl,axis=1),(N, 1)).T
        #dim = w.size
        Xn = -Lh.dot(Dm)/2.
        Xn -= np.tile(np.mean(Xn,axis=1),(N, 1)).T

        _, evecs = sc.linalg.eigh(Xn.dot(Xn.T))
        return (evecs[:,::-1].T.dot(Xn)).T
