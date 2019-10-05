## Implementation of K-Means clustering with random initialization.  ##
## Application to a toy example with n=1000, p=3, & K=3 is included. ##
##                                                                   ##
## Author: Greg Holste                                               ##
## Last Modified: 10/5/19                                            ##
#######################################################################

from sklearn.datasets import make_blobs
from sklearn import cluster
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


def KMeans(data, K, inits, max_iters=100):
    '''Implementation of K-Means clustering algorithm.

    Args:
        data (ndarray):  matrix of continuous data of shape (n, p)
        K (int):         desired number of clusters
        inits (int):     number of times to run K-Means (# random initializations of centroids)
        max_iters (int): maximum number of times to update centroids
    Returns:
        C (list):    encoder (cluster assignments) of "best" initialization
        SSE (list):  within-cluster scatter ("sum of squared errors") of best initialization
        M (ndarray): final centroids from best initialization of shape (K, p)
    '''
    np.random.seed(0)

    SSEs = []
    Cs = []
    Ms = []
    for _ in range(inits):

        # Randomly initalize K centroids/means
        m = []  # means
        for _ in range(K):
            m.append(np.random.uniform(low=[np.min(data[:, p]) for p in range(data.shape[1])],
                                       high=[np.max(data[:, p]) for p in range(data.shape[1])]
                                      )
                    )
        m = np.array(m)

        C_old = np.random.randint(0, K, size=data.shape[0])  # initialize encoder
        C_new = np.random.randint(0, K, size=data.shape[0])

        counter = 0
        while np.sum((C_new - C_old)**2) != 0 and counter < max_iters:  # while cluster assignments change...
            C_old = deepcopy(C_new)  # old becomes new from previous iteration

            # Encoder: assign observations to nearest centroid
            for i in range(data.shape[0]):
                dists = [np.sum((data[i, :] -  m[k])**2) for k in range(K)]
                C_new[i] = np.argmin(dists)


            # Find K means
            for k in range(K):
                idxs = np.where(C_new == k)[0]
                
                if idxs.size != 0:
                    m[k] = np.mean(data[idxs, :], axis=0)  # column-wise means

            counter += 1

        # Find within-cluster scatter ("sum of squared errors")
        SSE = 0.
        for i in range(data.shape[0]):
            SSE += np.sum((data[i, :] - m[C_new[i]])**2)
        
        # SSE = 0.
        # for k in range(K):
        #     idxs = np.where(C_new == k)[0]

        #     if idxs.size != 0:
        #         Nk = data[idxs, :].shape[0]  # num observations in cluster k
        #         SSE += Nk * np.sum((data[idxs, :] - m[k])**2) # or Nk/data.shape[0]

        SSEs.append(SSE)
        Cs.append(C_new)
        Ms.append(m)

    return Cs[np.argmin(SSEs)], SSEs[np.argmin(SSEs)], np.array(Ms[np.argmin(SSEs)])

