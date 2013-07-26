"""
Code provided by Danny Tarlow, Kevin Swersky, Laurent Charlin, Ilya Sutskever and Rich Zemel

Permission is granted for anyone to copy, use, modify, or distribute this
program and accompanying programs and documents for any purpose, provided
this copyright notice is retained and prominently displayed, along with
a note saying that the original programs are available from our 
web page.

The programs and documents are distributed without any warranty, express or
implied.  As the programs were written for research purposes only, they have
not been tested to the degree that would be advisable in any important
application.  All use of these programs is entirely at the user's own risk.

This code implements the methods described in the paper:
Stochastic k-neighborhood selection for supervised and unsupervised learning. ICML 2013.
"""

import numpy as np
import pylab
from scipy.spatial.distance import cdist
import knca_alg as ka

def plot_knca(X,y,A):
    """
    Plots the projected data (2D projections only).
    """
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    markers = ['o','s','d','p','*','+','<','^']

    j = 0
    Xp = np.dot(X,A.T)
    pylab.hold('on')
    for i in np.unique(y):
        c = colors[int(i % len(colors))]
        m = markers[int(i % len(markers))]
        pylab.plot(Xp[y==i,0],Xp[y==i,1],marker=m,color=c,linestyle='None')
        j += 1
    pylab.hold('off')

def make_kncas(k,kmin,N,Ccounts,C,batch_size):
    """
    Makes some objects to be used for performing inference.

    Args:
        k: The number of neighbors to consider for the knca objective.
        kmin: In a nutshell: set kmin=1 for the 'majority' version, and kmin=k
            for the 'all' version. Here's the more detailed explanation:
            The minimum number of points that must belong to the same class as 
            the target example within any given neighborhood set.
            That is, if I have k=5 then kmin=3 means that the objective will
            consider neighborhoods where 3 points belong to the same class as the
            target, as well as 4 and 5. If kmin=5 then only sets where all points
            are the same as the target class will be considered. If for exmaple
            k=5 and kmin=1 then it will ignore any cases where choosing that
            number of neighbors will not result in a majority belonging to the
            same target class. That is, if we have two classes then any set where only
            1 or 2 examples belong to the target class cannot be a majority, so
            these cases will be ignored. That is, in this example, setting
            kmin=1,2 or 3 will be equivalent (but this is not necessarily true
            if there are more than two classes).
        N: The size of the minibatch to be used for training.
        Ccounts: A vector of the counts of each class in the dataset.
        C: The number of classes
        batch_size: The size of each minibatch to be used for training.

    Returns:
        kncas: A list of KNCAAlg objects. Each will perform inference for a specific
            number of possible neighbors.
        knca0: A KNCAAlg object that will be used to compute the denominator of the knca
            distribution, which is a sum over all subsets of size k.
    """
    Ccounts = Ccounts.astype(np.double)
    kncas = []
    if (kmin != k):
        for k in range(kmin,int(k+1)/2):
            knca = ka.KNCAAlg(batch_size, N, k, k, Ccounts, C)
            kncas.append(knca)
        knca = ka.KNCAAlg(batch_size, N, k, -2, Ccounts, C)
        kncas.append(knca)
    else:
        knca = ka.KNCAAlg(batch_size, N, k, k, Ccounts, C)
        kncas.append(knca)

    knca0 = ka.KNCAAlg(batch_size, N, k, -1, Ccounts, C)

    return kncas,knca0

def run_knn(Xtest,Ytest,Xtrain,Ytrain,A,kmin=1,kmax=15):
    """
    Runs k-nearest neighbors on sum test data for a given range of k. Only performs knn for odd numbers.
    """
    acc = []
    Xtestp = np.dot(Xtest,A.T)
    Xtrainp = np.dot(Xtrain,A.T)
    print 'Calculating distances...'
    all_dists = -cdist(Xtestp,Xtrainp,'sqeuclidean')
    for k in range(kmin,kmax+1,2):
        print 'Running knn for k=%d' % k
        acc.append(knn(Xtest,Ytest,Xtrain,Ytrain,A,k,all_dists=all_dists))
    return np.array(acc)

def knn(Xtest,Ytest,Xtrain,Ytrain,A,K,all_dists=None):
    """
    Runs k-nearest neighbors on sum test data for a specific setting of k. Only advised for odd numbers.
    Optional args:
        all_dists: A pre-computed distance matrix between all pairs of points. This will be computed on the fly
            if this variable is None, which may be slow.
    """
    C = int(Ytrain.max()+1)

    num_errs = 0
    if (all_dists is None):
        Xtestp = np.dot(Xtest,A.T)
        Xtrainp = np.dot(Xtrain,A.T)
        print 'Calculating distances...'
        all_dists = -cdist(Xtestp,Xtrainp,'sqeuclidean')
    for i in range(Xtest.shape[0]):
        print 'Test point %d\r' % (i+1),
        dists = all_dists[i,:]
        if (id(Xtest) == id(Xtrain)):
            dists[i] = -np.inf
        cutoffs = np.argsort(dists)[-K:]
        class_votes = np.zeros(C)
        for k in range(C):
            class_votes[k] += np.sum(Ytrain[cutoffs]==k)
        num_errs += class_votes.argmax() != Ytest[i]
    print '\nFinished knn for k=%d.' % K
    return 1 - (num_errs / np.double(Ytest.shape[0]))

def split_data(seed,X,Y):
    """
    Used to split data into train/test sets using a given random seed.
    """
    np.random.seed(seed)
    p = np.random.permutation(X.shape[0])
    Xtrain = X[p[0:int(0.7*X.shape[0])],:]
    Xtest = X[p[int(0.7*X.shape[0]):],:]
    Ytrain = Y[p[0:int(0.7*X.shape[0])]]
    Ytest = Y[p[int(0.7*X.shape[0]):]]
    return Xtrain,Ytrain,Xtest,Ytest