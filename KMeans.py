"""Image Segmentation using K-means Clustering"""

from __future__ import print_function
from __future__ import division

import numpy as np
from scipy.spatial.distance import euclidean
import scipy.misc as spm
from matplotlib import pyplot
from sklearn.utils import check_random_state
from sklearn.utils import check_array

class KMeans():

    def __init__(self, n_clusters=3, max_iter=150, tol=1e-3,
                 verbose=0, random_state=None):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def _check_fit_data(self, X):
        X = check_array(X, dtype=np.float64)

        n_samples, n_features = X.shape
        if n_samples < self.n_clusters:
            raise ValueError("Number of samples="+str(n_samples)+" should be\
                greater than number of clusters="+str(self.n_clusters))
        return X

    def _update_centroids(self, X):
        for j in xrange(self.n_clusters):
            mask = self.labels_==j
            if np.sum(mask) == 0:
                raise ValueError("Empty Cluster")
            temp = X[mask]
            count = np.shape(temp)[0]
            self.centroids_[j] = np.sum(temp, axis=0)/count

    def _update_dist(self, X, dist):
        n_samples = X.shape[0]
        for j in xrange(n_samples):
            for k in xrange(self.n_clusters):
                cost = euclidean(X[j], self.centroids_[k])
                dist[j, k] = cost

    def fit(self, X, y=None, sample_weight=None):
        X = self._check_fit_data(X)

        if self.max_iter <= 0:
            raise ValueError("Maximum number of iterations must be greater \
                than zero")

        n_samples, n_features=X.shape

        rs = check_random_state(self.random_state)
        self.labels_ = np.zeros((n_samples))
        centroids_idx = rs.randint(n_samples, size=self.n_clusters)
        self.centroids_ = X[centroids_idx]

        dist = np.zeros((n_samples, self.n_clusters))

        for itr in xrange(self.max_iter):
            dist.fill(0)
            self._update_dist(X, dist)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            n_same = np.sum(self.labels_ == labels_old)
            if 1-n_same/n_samples < self.tol:
                if self.verbose:
                    print("Converged at iteration "+ str(itr+1))
                break
            self._update_centroids(X)

        self.X_fit_ = X
        return self
        
# 0 -> black

def segmentImage():
    img = spm.lena()
    img = spm.imresize(img, (300, 300))
    height, width = img.shape
    clf = KMeans(n_clusters=3, max_iter=100, random_state=0, verbose=1)
    img_list = np.reshape(img, (height*width, 1))
    clf.fit(img_list)
    index = np.copy(clf.labels_)
    index = np.reshape(index, (height, width))
    axes = pyplot.gca()
    axes.imshow(index)
    pyplot.show(block=True)

if __name__=='__main__':
    segmentImage()
