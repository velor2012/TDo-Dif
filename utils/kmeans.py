
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn import datasets
from sklearn.metrics import pairwise_distances
import heapq
def make_clusters(skew=True, *arg,**kwargs):
    X, y = datasets.make_blobs(*arg,**kwargs)
    if skew:
        nrow = X.shape[1]
        for i in np.unique(y):
            X[y==i] = X[y==i].dot(np.random.random((nrow,nrow))-0.5)
    return X,y

def scatter(X, color=None, ax=None, centroids=None):
    assert X.shape[1]==2
    if color is not None:
        assert X.shape[0]==color.shape[0]
        assert len(color.shape)==1
    if not ax:
        _, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=color)
    if centroids is not None:
        ax.scatter(centroids[:,0],centroids[::,1], marker="o",s=350, c=range(centroids.shape[0]))

# assign to the clusters (E-step)
def get_assignments(X, centroids,weight=None,proportion=None):
    dist = pairwise_distances(X, centroids,metric='euclidean') #dist为NXK
    # weight = np.random.random(len(dist))-0.5
    if weight is not None:
        weight = weight.reshape(-1,1)
        dist = dist * weight
    assign = np.argmin(dist,axis=1)
    if proportion is not None:
        query_num = int(len(X) * proportion)
        min_distances = np.min(dist,axis=1)
        valid_sample_index = np.argsort(min_distances)[:query_num]
        
        # print(min_distances[valid_sample_index])
        return assign,valid_sample_index
    return assign


# compute the new centroids (M-step)
def get_centroids(X, assignments):
    centroids = []
    for i in np.unique(assignments):
        centroids.append(X[assignments==i].mean(axis=0))     
    return np.array(centroids)

def KMeans(X, centroids, n_iterations=100, axes=None,weight=None,proportion=None):
    if axes is not None:
        axes = axes.flatten()
    last_centroids = centroids
    for i in range(n_iterations):
        res = get_assignments(X, centroids,weight =weight,proportion=proportion)
        if proportion is not None:
            assignments = res[0]
            active_samples = res[1]
        else:
            assignments = res
        centroids = get_centroids(X, assignments)
        if axes is not None:
            scatter(X, assignments, ax=axes[i], centroids=centroids)
            axes[i].set_title(i)
        # 2c. Check for convergence
        if np.all(centroids == last_centroids):
            break
        else:
            last_centroids = centroids
        if proportion is not None:
            return assignments, centroids,active_samples
    return assignments, centroids


# initize the centroids
def init_random(X, K):
    center_index = np.random.choice(range(X.shape[0]), K)
    centroids = X[center_index,:]
    return centroids

def init_kmeans_plus_plus(X, K):
    '''Choose the next centroids with a prior of distance.'''
    assert K>=2, "So you want to make 1 cluster?"
    compute_distance = lambda X, c: pairwise_distances(X, c).min(axis=1)
    # get the first centroid
    centroids = [X[np.random.choice(range(X.shape[0])),:]]
    # choice next
    for _ in range(K-1):
        proba = compute_distance(X,centroids)**2
        proba /= proba.sum()
        centroids.append(X[np.random.choice(range(X.shape[0]), p=proba)])      
    return np.array(centroids)

def my_KMeans_plusplus(X,K,weight=None,proportion=None):
    centroids = init_kmeans_plus_plus(X, K)
    y, centroids,val_samples = KMeans(X, centroids,weight=weight,proportion=proportion)
    return y,centroids,val_samples

if __name__ == '__main__':
    K = 3
    X, y = make_clusters(skew=False, n_samples=1500, centers=K, cluster_std=[.5,.5,.8])
    answer = y
    # kmeans++ initization is much better (the centroids are far from each other), which leads to better converge.
    # Note: kmeans++ may still lead to local minimum
    np.random.seed(1001)
    n_tests = 10
    _, axes = plt.subplots(10,4, figsize=(12,3*n_tests))
    axes[0,0].set_title('Randomly initized')
    axes[0,1].set_title('Randomly initized Converged')
    axes[0,2].set_title('KMeans++ initized')
    axes[0,3].set_title('KMeans++ initized Converged')


    for i in range(n_tests):
        centroids = init_random(X, K)
        scatter(X, answer, ax=axes[i,0], centroids=centroids)
        y, centroids = KMeans(X, centroids)
        scatter(X, y, ax=axes[i,1], centroids=centroids)

        centroids = init_kmeans_plus_plus(X, K)
        scatter(X, answer, ax=axes[i,2], centroids=centroids)
        y, centroids = KMeans(X, centroids)
        scatter(X, y, ax=axes[i,3], centroids=centroids)
    plt.savefig('test.png')