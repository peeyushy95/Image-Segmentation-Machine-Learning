"""Image Segmentation using Filtering algorithm"""

from __future__ import print_function
from __future__ import division
import time
import numpy as np
import scipy.misc as spm
from scipy.spatial.distance import euclidean
from matplotlib import pyplot
from sklearn.utils import check_random_state
from sklearn.utils import check_array

class Filtering():

    def __init__(self, n_clusters=3, max_iter=150, tol=1e-3):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def data_check(self, X):
        
        X = check_array(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        if n_samples < self.n_clusters:
            raise ValueError("Number of samples should be greater than number of clusters")
        
        return X

    def update_centroids(self, X):
       
       for j in xrange(self.n_clusters):
            clus=0
            mask = self.labels==j
            
            if np.sum(mask) == 0:
                clus+=1
                #raise ValueError("Empty Cluster")
            temp = X[mask]
            count = np.shape(temp)[0]+clus
            self.centroids[j] = np.sum(temp, axis=0)/count
            

    def update_dist(self, X):
        n_samples = X.shape[0]
        for j in xrange(n_samples):
            
            cost = euclidean(X[j], self.centroids[0])
            ind = 0;
            
            for k in range(1,self.n_clusters):
                
                temp = euclidean(X[j], self.centroids[k])
                if(temp < cost):
                    cost = temp
                    ind = k
                    
            self.labels[j] = ind    

    def filter(self,X, start, end, centroids):
        
        mean = 0.0
        for i in range(start,end+1):
            mean += X[i];
        mean/=(end - start + 1)
        
        z_ = centroids[0]
        ind = 0
        
        cost = euclidean(mean, centroids[0])
        
        for i in range(1,centroids.size):
            print(mean,centroids[i])
            print(i,centroids.size)
            temp = euclidean(mean, centroids[i])
            if(temp < cost):
                cost = temp
                ind = i
        
        z_ = ind
                
        if start == end :
            self.labels[start] = z_
        else :
    
            if  (centroids.size) == 1:
                for i in range(start,end+1):
                    self.labels[i] = z_
            else :
                x = np.array([])
                for i in range (centroids.size):
                    count = 0
                    for j in range(start,end+1):
                        if euclidean(X[j],centroids[i]) <=  euclidean(X[j],centroids[z_]):
                            count+=1
                    if(count >= (end - start + 1)):
                        x=np.append(x,centroids[i])
                        
                self.filter(X,start,(start+end)//2,x)
                self.filter(X,(start+end)//2 + 1,end,x)
                    
                
        
        
    def initiate(self, X):
        
        X = self.data_check(X)

        if self.max_iter <= 0:
            raise ValueError("Maximum number of iterations must be greater than zero")

        n_samples, n_features=X.shape

        rs = check_random_state(0)        
        centroids_idx = rs.randint(n_samples, size=self.n_clusters)
        
        self.centroids = X[centroids_idx]
        self.labels = np.zeros((n_samples))
        labels_old  = np.zeros((n_samples))
    
        for itr in xrange(self.max_iter):
            
            for i in range(n_samples):
                labels_old[i] = self.labels[i]
                
            self.filter(X,0,n_samples-1,self.centroids);
            #self.update_dist(X)
          
            n_same = 0
            for i in range(n_samples):
                if labels_old[i] == self.labels[i]:
                    n_same+=1
            
            if 1-n_same/n_samples < self.tol:
                print("Converged at iteration "+ str(itr+1))
                break
            
            self.update_centroids(X)

        self.X_fit_ = X
        return self
        
# 0 -> black

def segmentImage():
    
    img = spm.lena()
    img = spm.imresize(img, (200, 200))
    height, width = img.shape
    
    obj = Filtering(n_clusters=3, max_iter=100 )
    img_list = np.reshape(img, (height*width, 1))
    obj.initiate(img_list)
    index = np.copy(obj.labels)
    index = np.reshape(index, (height, width))
    axes = pyplot.gca()
    axes.imshow(index)
    pyplot.show(block=True)

if __name__=='__main__':
    start_time = time.clock()
    segmentImage()
    print (time.clock() - start_time, "seconds")
