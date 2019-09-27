## Implementation of K-Means clustering with random initialization.  ##
## Application to a toy example with n=1000, p=3, & K=3 is included. ##
##                                                                   ##
## Author: Greg Holste                                               ##
## Last Modified: 9/25/19                                            ##
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


# def main():
#     # Create toy dataset with K=3 clusters
#     X, y = make_blobs(n_samples=1000, n_features=2, centers=3, random_state=170)

#     # Run K-Means with K=3
#     C, SSE, M = KMeans(data=X, K=3, inits=20)

#     new_SSE = 0.
#     for k in range(3):
#         idxs = np.where(C == k)[0]

#         if idxs.size != 0:
#             Nk = X[idxs, :].shape[0]  # num observations in cluster k
#             new_SSE += Nk * np.sum((X[idxs, :] - M[k])**2) # or Nk/data.shape[0]


#     # Run sklearn K-Means with K = 3
#     sk = cluster.KMeans(n_clusters=3, init="random", n_init=20,
#                           random_state=0, max_iter=100).fit(X)

#     new_sk_SSE = 0.
#     for k in range(3):
#         idxs = np.where(sk.labels_ == k)[0]

#         if idxs.size != 0:
#             Nk = X[idxs, :].shape[0]  # num observations in cluster k
#             new_sk_SSE += Nk * np.sum((X[idxs, :] - sk.cluster_centers_[k])**2) # or Nk/data.shape[0]


#     fig, ax =  plt.subplots(1, 3, figsize=(12,6))
#     ax[0].scatter(X[:, 0], X[:, 1], c=y)
#     ax[0].set_title("Original labeled data")

#     ax[1].scatter(X[:, 0], X[:, 1], c=sk.labels_, alpha=0.2)
#     ax[1].scatter(sk.cluster_centers_[:, 0], sk.cluster_centers_[:, 1], marker="*", s=200, c="black")
#     ax[1].set_title("Sklearn Clustered Data (K=3)")
#     ax[1].set_xlabel(f"Within-Cluster Scatter: {round(sk.inertia_, 3)} \n Weighted W(C): {round(new_sk_SSE, 3)}")

#     ax[2].scatter(X[:, 0], X[:, 1], c=C, alpha=0.2)
#     ax[2].scatter(M[:, 0], M[:, 1], marker="*", s=200, c="black")
#     ax[2].set_title("Custom Clustered Data (K=3)")
#     ax[2].set_xlabel(f"Within-Cluster Scatter: {round(SSE, 3)} \n Weighted W(C): {round(new_SSE, 3)}")

#     plt.show()

    # # Run K-means for K=1,...,6 and make "elbow plot"
    # SSEs = []
    # for k in range(1, 7):
    #     _, SSE, _ = KMeans(data=X, K=k, inits=5)
    #     SSEs.append(SSE)

    # fig, ax = plt.subplots(1, 1, figsize=(6,6))
    # ax.scatter(list(range(1, 7)), SSEs)
    # ax.set_xlabel("K")
    # ax.set_ylabel("Within-Cluster Scatter")
    # plt.show()


# main()




# ######
# import pandas as pd
# import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D

# def normalize(vec):
#   return (vec - np.mean(vec)) / np.std(vec)


# data = pd.read_csv("Data/CWR_trait_data.csv")
# X = data.drop("Species", axis=1).to_numpy()
# print(X)


# print(np.mean(X[:, 0]), np.std(X[:, 0]))
# print(np.mean(X[:, 1]), np.std(X[:, 1]))
# print(np.mean(X[:, 2]), np.std(X[:, 2]))

# # for i in range(X.shape[1]):
# #     X[:, i] = normalize(X[:, i])

# print(np.mean(X[:, 0]), np.std(X[:, 0]))
# print(np.mean(X[:, 1]), np.std(X[:, 1]))
# print(np.mean(X[:, 2]), np.std(X[:, 2]))

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.5)
# ax.set_xlabel(data.columns.values[1])
# ax.set_ylabel(data.columns.values[2])
# ax.set_zlabel(data.columns.values[3])
# ax.set_title(f"CWR Trait Data (n={X.shape[0]})")
# for i, txt in enumerate(data["Species"].tolist()):
#   ax.text(X[i, 0], X[i, 1], X[i, 2], txt, 'y', fontsize=6, alpha=0.5)

# plt.show()


# Ms = []
# Cs = []
# SSEs = []
# for k in range(1, 8):
#   C, SSE, M = KMeans(data=X, K=k, inits=30)
    
#   Cs.append(C)
#   SSEs.append(SSE)
#   Ms.append(M)

# Ms = np.array(Ms)

# sk_Ms = []
# sk_Cs = []
# sk_SSEs = []
# for k in range(1, 8):
#     sk = cluster.KMeans(n_clusters=k, init="random", n_init=30,
#                         random_state=0, max_iter=100).fit(X)
    
#     sk_Ms.append(sk.cluster_centers_)
#     sk_Cs.append(sk.labels_)
#     sk_SSEs.append(sk.inertia_)


# # Elbow plot
# # ax = sns.scatterplot(list(range(1, 8)), SSEs)
# # ax.set(xlabel = "K", ylabel = "Within-Cluster Scatter")
# fig, ax = plt.subplots(1, 2, figsize=(12,6))
# ax[0].scatter(list(range(1, 8)), SSEs)
# ax[1].scatter(list(range(1, 8)), sk_SSEs)

# plt.show()

# # K=2 results
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.5, c=Cs[1])
# ax.scatter(Ms[1][:, 0], Ms[1][:, 1], Ms[1][:, 2], marker="*", s=200, c=[0,1])
# ax.set_xlabel(data.columns.values[1])
# ax.set_ylabel(data.columns.values[2])
# ax.set_zlabel(data.columns.values[3])
# ax.set_title("Clustered Trait Data (K=2)")
# plt.show()

# # K=3 results
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.5, c=Cs[2])
# ax.scatter(Ms[2][:, 0], Ms[2][:, 1], Ms[2][:, 2], marker="*", s=200, c=[0,1,2])
# ax.set_xlabel(data.columns.values[1])
# ax.set_ylabel(data.columns.values[2])
# ax.set_zlabel(data.columns.values[3])
# ax.set_title("Clustered Trait Data (K=3)")
# plt.show()

# # K=4 results
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.5, c=Cs[3])
# ax.scatter(Ms[3][:, 0], Ms[3][:, 1], Ms[3][:, 2], marker="*", s=200, c=[0,1,2,3])
# ax.set_xlabel(data.columns.values[1])
# ax.set_ylabel(data.columns.values[2])
# ax.set_zlabel(data.columns.values[3])
# ax.set_title("Clustered Trait Data (K=4)")
# plt.show()




