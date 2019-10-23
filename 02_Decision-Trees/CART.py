## Simple implementation of Classification and Regression Trees (CART).     ##
## I'm not entirely sure it works correctly for regression trees...         ##
## Inspiration taken from Google Developers project https://bit.ly/2CLZpNj. ##
##                                                                          ##
## Author: Greg Holste                                                      ##
## Last Modified: 10/23/19                                                  ##
##############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter


def RMSE(df):
    # Find root mean squared error for samples in 'df'
    N = df.shape[0]
    y_pred = np.mean(df.iloc[:, -1])
    y_true = df.iloc[:, -1]

    return (np.mean((y_true - y_pred)**2))**(1/2)

def Gini(df):
    # Find Gini impurity for samples in 'df'
    N = df.shape[0]
    Q = 0.
    for c in np.unique(df.iloc[:, -1]):
        p_hat = np.sum(df.iloc[:, -1] == c) / N
        Q += p_hat * (1 - p_hat)

    return Q

class LeafNode:
    # Terminal ("leaf") node of decision tree
    def __init__(self, df, y_type):
        if y_type == 'discrete':
            label_counts = Counter(df.iloc[:, -1])
            self.class_probs = {k: v / df.shape[0] for k, v in label_counts.items()}
            self.pred = list(self.class_probs.keys())[0]
            self.prob = list(self.class_probs.values())[0]
        else:
            self.pred = np.mean(df.iloc[:, -1])

class DecisionNode:
    # Nonterminal ("decision") node of decision tree
    def __init__(self, J, S, tL, tR):
        self.J = J
        self.S = S
        self.tL = tL
        self.tR = tR

class DecisionTree:
    '''Class to fit CART-style decision tree to data with continuous features.
    
    Args:
        data (pandas DataFrame): data frame of shape (n x (p+1)), where the first p
                                 columns are continuous predictors and the LAST column
                                 is the response variable (continuous or discrete)
        y_type (string)        : string denoting whether to build tree for regression or classification
        min_node_size (int)    : minimum terminal node size allowed before ending the tree-growing process

    Attributes:
        data (pandas DataFrame): data frame of shape (n x (p+1)), where the first p
                                 columns are continuous predictors and the LAST column
                                 is the response variable (continuous or discrete)
        y_type (string)        : string denoting whether to build tree for regression or classification
        min_node_size (int)    : minimum terminal node size allowed before ending the tree-growing process
        num_leaves (int)       : TODO -- number of terminal nodes in fitted tree
        depth (int)            : TODO -- number of "levels" of fitted tree (from root to deepest leaf)
    '''
    def __init__(self, data, y_type='discrete', min_node_size=1):
        assert (y_type in ['discrete', 'continuous']), "y_type must be 'discrete' or 'continuous'"
        self.y_type = y_type

        if self.y_type == 'discrete':
            self.Q = Gini
        else:
            self.Q = RMSE

        self.c = min_node_size
        self.root_node = self.fit(data)
        self.num_leaves = 0  # TODO
        self.depth = 0       # TODO

    def fit(self, df):
        '''Function to grow a decision tree for classification or regression.

        Args:
            df (pandas DataFrame): data frame of shape (? x (p+1)), where the first p
                                   columns are continuous predictors and the LAST column
                                   is the response variable (continuous or discrete)
        Returns: 
            root_node (DecisionNode): reference to root node of trained decision tree
        '''
        N = df.shape[0]

        criteria = []
        Js = []
        Ss = []
        tLs = []
        tRs = []
        for j in range(df.shape[1] - 1):
            for s in np.unique(df.iloc[:, j]):
                tL = df[df.iloc[:, j] <= s]
                tR = df[df.iloc[:, j] > s]

                QL = self.Q(tL)
                QR = self.Q(tR)
                wsum = (tL.shape[0]/N)*QL + (tR.shape[0]/N)*QR  # opt. criterion

                criteria.append(wsum)
                Js.append(j)
                Ss.append(s)
                tLs.append(tL)
                tRs.append(tR)

                # print(f"X{j+1} <= {s}: {wsum} | QL: {QL}, QR: {QR}, Q: {self.Q(df)}")

        criteria = np.array(criteria)
        tL_sizes = np.array([t.shape[0] for t in tLs])
        tR_sizes = np.array([t.shape[0] for t in tRs])

        elig_idxs = np.intersect1d( np.where(tL_sizes >= self.c), np.where(tR_sizes >= self.c) )
        if elig_idxs.size == 0 or self.Q(df) == 0:
            return LeafNode(df, y_type=self.y_type)
        else:
            idx = elig_idxs[np.argmin(criteria[elig_idxs])]
            criterion = criteria[idx]
            J = Js[idx]
            S = Ss[idx]
            tL = tLs[idx]
            tR = tRs[idx]

            left_child = self.fit(tL)
            right_child = self.fit(tR)

        return DecisionNode(J, S, left_child, right_child)

    def print(self):
        def print_tree(node, spacing=""):
            if isinstance(node, LeafNode):
                if self.y_type == 'discrete':
                    print(spacing + "Predict", node.pred, f"({round(node.prob*100)}%)")
                else:
                    print(spacing + "Predict", node.pred)
                return

            print(spacing + f"X{node.J + 1} <= {node.S}?")

            print(spacing + "--> True:")
            print_tree(node.tL, spacing + "  ")

            print(spacing + "--> False:")
            print_tree(node.tR, spacing + "  ")

        print_tree(self.root_node)

    def predict(self, sample):
        def predict_tree(sample, node):
            if isinstance(node, LeafNode):
                return node.pred

            if sample[node.J] <= node.S:
                return predict_tree(sample, node.tL)
            else:
                return predict_tree(sample, node.tR)

        return predict_tree(sample, self.root_node)
