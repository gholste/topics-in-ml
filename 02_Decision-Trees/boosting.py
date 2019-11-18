## Simple implementation of two boosting techniques: AdaBoost (binary classification) ##
## and Gradient Boosted Regression Trees (regression) using sklearn decision trees.   ##
##                                                                                    ##
## Author: Greg Holste                                                                ##
## Last Modified: 11/17/19                                                            ##
########################################################################################

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from copy import deepcopy

class AdaBoost:
    '''Implementation of AdaBoost.M1 algorithm for binary classification.

    Args:
        X (ndarray)    : (n x p) matrix of features
        y (ndarray)    : (n) or (n x 1) array of labels in the set {-1, 1}
        M (int)        : number of learners (trees) in boosted model
        J (int)        : "depth" of each tree in boosted model

    Attributes:
        X (ndarray)    : (n x p) matrix of features
        y (ndarray)    : (n) or (n x 1) array of labels in the set {-1, 1}
        M (int)        : number of learners (trees) in boosted model
        J (int)        : "depth" of each tree in boosted model
        G (list)       : list of M learners in ensemble [AFTER CALL TO FIT]
        Alpha (list)   : list of M learner weights in final output [AFTER CALL TO FIT]
        trained (bool) : whether or not model has been trained [AFTER CALL TO FIT]
    '''
    def __init__(self, X, y, M, J=1):
        assert (np.all(sorted(np.unique(y)) == [-1, 1])), "Labels must all be 1 or -1"
        self.X = X
        self.y = y
        self.M = M
        self.J = J

        self.G = []
        self.Alpha = []
        self.trained = False

    def fit(self):
        ## CHECK X AND Y ARE PROPER ## 
        W = np.ones(self.X.shape[0]) * (1 / self.X.shape[0])  # initialize weights to 1/n

        for _ in range(self.M):
            DT = DecisionTreeClassifier(max_depth=self.J).fit(self.X, self.y, sample_weight=W)

            err_idxs = np.where(DT.predict(self.X) != self.y)[0]
            if err_idxs.size == 0:
                err = 1e-7
            else:
                err = np.sum(W[err_idxs]) / np.sum(W)

            alpha = np.log((1 - err) / err)

            W[err_idxs] *= np.exp(alpha)

            self.G.append(DT)
            self.Alpha.append(alpha)

        self.Alpha = np.array(self.Alpha)
        self.trained = True

    def predict(self, X):
        G_hats = np.array([g.predict(X) for g in self.G])

        return np.array([np.sign(np.sum(self.Alpha * G_hats[:, i])) for i in range(X.shape[0])])


class GradientBoostRegressor:
    '''Implementation of Gradient Boosted Regression Trees for regression.

    Args:
        X (ndarray)    : (n x p) matrix of features
        y (ndarray)    : (n) or (n x 1) array of continous labels
        M (int)        : number of trees in boosted model
        J (int)        : "depth" of each tree in boosted model
        lam (float)    : learning rate

    Attributes:
        X (ndarray)    : (n x p) matrix of features
        y (ndarray)    : (n) or (n x 1) array of continuous labels
        M (int)        : number of learners (trees) in boosted model
        J (int)        : "depth" of each tree in boosted model
        lam (float)    : learning rate
        F (list)       : list of M + 1 learners (including base model) [AFTER CALL TO FIT]
        trained (bool) : whether or not model has been trained [AFTER CALL TO FIT]
    '''
    def __init__(self, X, y, M, J=1, lam=0.01):
        assert (y.dtype not in ["int", "np.int32", "np.int64"]), "Labels must be real-valued (not int)"
        self.X = X
        self.y = y
        self.M = M
        self.J = J
        self.lam = lam

        self.F = [DecisionTreeRegressor(max_depth=1).fit(self.X, self.y)]
        self.trained = False

    def fit(self):
        y_pred = self.F[0].predict(self.X)

        for _ in range(self.M):
            resids = self.y - y_pred

            f_m = DecisionTreeRegressor(max_depth=self.J).fit(self.X, resids)

            y_pred += self.lam * f_m.predict(self.X)

            self.F.append(f_m)      

        self.F = np.array(self.F)
        self.trained = True

    def predict(self, X):
        return self.F[0].predict(X) + self.lam * np.sum([f.predict(X) for f in self.F[1:]], axis=0)

# def GradientBoostClassifier(X, y, M, J=1, alpha=0.01):
#   F = []  # at end, list of M lists of C decision trees

#   N = y.shape[0]
#   classes = sorted(np.unique(y))

#   orig_preds = np.zeros((N, len(classes)))
#   orig_preds[:, np.argmax([np.sum(c) for c in classes])] = 1.
#   y_preds = deepcopy(orig_preds)

#   for _ in range(M):
#       p = np.apply_along_axis(softmax, 1, y_preds)

#       Fm = []
#       for i, c in enumerate(classes):
#           resids = np.where(y == c, 1, 0) - p[:, i]

#           f_m = DecisionTreeRegressor(max_depth=J).fit(X, resids)
#           Fm.append(f_m)

#           y_preds[:, i] += alpha * f_m.predict(X)
#       F.append(Fm)

#   def classifier(x):
#       n = x.shape[0]
#       res = np.apply_along_axis(softmax, 1, orig_preds)
#       res = res[:n, ...]
#       print(res.shape)

#       preds = np.zeros((n, len(classes), M))
#       for c in range(len(classes)):
#           for m in range(M):
#               preds[:, c, m] = F[m][c].predict(x)

#       res += np.sum(preds, axis=-1)
#       res *= alpha

#       return np.array([classes[i] for i in np.argmax(res, axis=-1)])


#   return classifier

# def softmax(f):
#   denom = np.sum(np.exp(f))

#   return np.exp(f) / denom

def acc(y, y_hat):
    # Calculate mean accuracy given true and predicted discrete labels
    return np.sum(y == y_hat) / y.size

def R2(y, y_hat):
    # Calculate coefficient of determination (R^2) given true and predicted continuous labels
    u = np.sum((y - y_hat)**2)
    v = np.sum((y - np.mean(y))**2)

    return 1 - u / v
