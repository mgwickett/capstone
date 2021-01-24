
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import matplotlib 
import numpy as np
from numpy import nanmean 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
# Load image
import imageio


# Set random seed so output is all same
np.random.seed(1)


class KMeans(object):

    def __init__(self):  # No need to implement
        pass

    def pairwise_dist(self, x, y):  # [5 pts]
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between 
                x[i, :] and y[j, :]
                """
        # need dimensions of NMD
        
        dist = np.linalg.norm(x[:, np.newaxis] - y, axis = 2)
        return dist
        

    def _init_centers(self, points, K, **kwargs):  # [5 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers. 
        """
        # raise NotImplementedError
        rows = points.shape[0]
        indices = np.random.choice(rows, size = K, replace = True)
        centers = points[indices, :]
        return centers 

    def _update_assignment(self, centers, points):  # [10 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point

        Hint: You could call pairwise_dist() function.
        """
        # raise NotImplementedError
        something = self.pairwise_dist(points, centers)
        return np.argmin(something, axis = 1)
        

    def _update_centers(self, old_centers, cluster_idx, points):  # [10 pts]
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, K x D numpy array, where K is the number of clusters, and D is the dimension.
        """
        # raise NotImplementedError
        rows = old_centers.shape[0]
        cols = old_centers.shape[1]
        centers = np.empty((rows, cols))
        for k in range(0, rows):
            kArr = points[cluster_idx == k, :]
            if (kArr.any()):
                centers[k, :] = np.mean(kArr, 0)
            else:
                centers[k, :] = old_centers[k, :]
        return centers

    def _get_loss(self, centers, cluster_idx, points):  # [5 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans. 
        """
        # raise NotImplementedError
        k = centers.shape[0]
        distances = self.pairwise_dist(centers, points) # returns KxN numpy array 
        sigma = 0

        for i in range(0, k):
            distos = distances[i, cluster_idx == i]
            sigma += np.sum(np.square(distos))
        return sigma

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        return cluster_idx, centers, loss

    def find_optimal_num_clusters(self, data, max_K=15):  # [10 pts]
        """Plots loss values for different number of clusters in K-Means

        Args:
            image: input image of shape(H, W, 3)
            max_K: number of clusters
        Return:
            None (plot loss values against number of clusters)
        """

        # raise NotImplementedError
        ks = []
        losses = []
        for num_K in range(1, max_K + 1):
            _, _, loss = KMeans()(data, num_K)
            ks.append(num_K)
            losses.append(loss)
        plt.plot(ks, losses)
        return losses  
        

def intra_cluster_dist(cluster_idx, data, labels):  # [4 pts]
    """
    Calculates the average distance from a point to other points within the same cluster

    Args:
        cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        intra_dist_cluster: 1D array where the i_th entry denotes the average distance from point i 
                            in cluster denoted by cluster_idx to other points within the same cluster
    """
    # raise NotImplementedError
    points = data[labels == cluster_idx, :]
    dists = KMeans().pairwise_dist(points, points)
    dists[dists == 0] = np.nan
    intra_dist_cluster = np.nanmean(dists, axis = 1)
    return intra_dist_cluster


def inter_cluster_dist(cluster_idx, data, labels):  # [4 pts]
    """
    Calculates the average distance from one cluster to the nearest cluster
    Args:
        cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        inter_dist_cluster: 1D array where the i-th entry denotes the average distance from point i in cluster
                            denoted by cluster_idx to the nearest neighboring cluster
    """
    # raise NotImplementedError
    k = np.unique(labels).shape[0]
    main_cluster = data[labels == cluster_idx, :]
    main_size = main_cluster.shape[0]
    min_dist = np.full(main_size, np.inf)
    
    for i in range(k):
        if (i == cluster_idx):
            continue 
        k_cluster = data[labels == i, :]
        if (not k_cluster.any()):
            continue 
        distos = KMeans().pairwise_dist(main_cluster, k_cluster)
        avgDist = np.mean(distos, axis = 1)
        min_dist = np.minimum(min_dist, avgDist)
                
    return min_dist


def silhouette_coefficient(data, labels):  # [2 pts]
    """
    Finds the silhouette coefficient of the current cluster assignment

    Args:
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        silhouette_coefficient: Silhouette coefficient of the current cluster assignment
    """
    # raise NotImplementedError
    # uses intra and inter clust distance 
    # some equation somewhere 
    # a = intra, b = inter 
    # b(i) - a(i) / max(a(i), b(i))
    k = np.unique(labels).shape[0]
    n = data.shape[0]
    sc = 0 

    for i in range(k): 
        intra = intra_cluster_dist(i, data, labels)
        inter = inter_cluster_dist(i, data, labels)
        temp = (inter - intra) / (np.maximum(inter, intra))
        sc += np.sum(temp)

    silhouette_coefficient = sc / n
    return silhouette_coefficient
