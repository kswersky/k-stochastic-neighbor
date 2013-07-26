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

import sys
import numpy as np
from scipy.optimize import fmin_l_bfgs_b as lbfgs
import time
import pickle
import knca_alg as ka
import knca_utils as ku
import pylab
import usps

try:
    import matplotlib.pylab as plt
except:
    pass

import pickle
from scipy.spatial.distance import pdist, squareform, cdist
import scipy.cluster.hierarchy as hac
import pylab
from matplotlib.lines import Line2D

def demo_knca(seed=1,noise=0):
    print 'Loading USPS with seed %d and noise %s...' % (seed,noise)
    D = usps.get_usps_split_corrupt(seed,noise=0)
    Xtrain = D['Xtrain']
    Xtest = D['Xtest']
    ytrain = D['ytrain']
    ytest = D['ytest']

    P = Xtrain.shape[1]
    k = 10
    kmin = 10 # Use the 'all' version of knca
    num_iters=10
    batch_size=100
    eta=0.01
    mo=0.9


    print 'Training kNCA for %d iterations.' % num_iters
    A,obj_hist,test_acc_history = train_sgd(
            Xtrain,
            ytrain,
            P,
            k,
            kmin=kmin,
            num_iters=num_iters,
            batch_size=batch_size,
            eta=eta,
            mo=mo,
            Xtest=Xtest,
            ytest=ytest)

    print 'Finished training.'
    print 'Objective history: %s' % obj_hist
    print 'Test accuracy history: %s' % test_acc_history

    if P == 2:
        print 'Plotting projected data...'
        ku.plot_knca(Xtest,ytest,A)
        pylab.ioff()
        pylab.show()

def train_sgd(X,y,P,k,
        kmin=1,
        A=None,
        num_iters=1,
        batch_size=10,
        eta=0.1,
        mo=0,
        Xtest=None,
        ytest=None,
        test_kmin=1,
        test_kmax=15):

    """Trains KNCA using stochastic gradient descent.

    Given a full dataset X and associated integer labels Y (from 0 to #classes-1),
    this function will train a projection matrix A such that dot(A.T,A) forms a
    Mahalanobis distance metric in order to make k-nearest-neighbors work better
    in the projected space Z = dot(A,X).

    The training algorithm is stochastic gradient descent (SGD), where minibatches
    consisting of rows of X are chosen at random (without replacement) at
    each iteration (a.k.a. epoch) of learning. Each point still considers the
    full set of point when marginalizing over neighbor sets.

    Note that one iteration of learning is fundamentally O(N^2), so this
    is not expected to scale gracefully with dataset size. Also beware of numerical
    issues that may arise from the underlying message passing algorithm. In
    the future this may be replaced with a slightly slower, yet more stable
    algorithm that passes messages in log-space.

    Args:
        X: The training data where each row corresponds to an example.
        y: A vector of labels from 0 to #classes - 1.
        P: The dimension of the projection (X.shape[1] corresponds to full-dimensional).
        k: The number of neighbors to consider for the knca objective.

    Optional Args:
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
        num_iters: The number of times to cycle through the dataset for SGD.
        batch_size: The number of examples to consider per parameter update for SGD.
        eta: The SGD learning rate.
        mo: The SGD momentum (each update direction uses a bit of the old
            update direction to prevent oscillations). Can make learning faster,
            0.9 can give good results but probably not wise to go beyond that.
        Xtest: Test data (include this to get a test-error history).
        ytest: Test labels (include this to get a test-error history).
        test_kmin: The minimum value of k for running knn on the test set.
        test_kmax: The maximum value of k for running knn on the test set.
            Note, only odd values between test_kmin and test_kmax are evaluated.

    Returns:
        A: The projection matrix.
        obj_hist: A vector corresponding to the approximate training objective
            after each pass through the data.
        test_err_hist: A vector corresponding to the KNN test accuracy after each
            pass through the data.
    """

    N = X.shape[0]   # num points
    D = X.shape[1]   # input dimension
    C = int(y.max()+1)    # num classes

    num_batches = np.ceil(np.double(N)/batch_size)

    """
    Initialize A using relevant component analysis
    (A. Bar-Hillel, T. Hertz, N. Shental, and D. Weinshall.
    Learning distance functions using equivalence relation. ICML 2003.)
    Random initialization would probably work just fine too.
    """
    if (A is None):
        A = np.random.randn(P,D)
        [U,S,V] = np.linalg.svd(A,full_matrices=False)
        S = np.ones(S.shape)
        A = np.dot(U,np.dot(np.diag(S),V))
    else:
        A = A.copy()

    #Makes some objects to be used for performing inference.
    Ccounts = np.array([np.sum(y == c) for c in range(C)])
    kncas,knca0 = ku.make_kncas(k,kmin,N,Ccounts,C,batch_size)

    #Run SGD
    dA = 0
    obj_hist = []
    test_acc_history = []
    for i in range(num_iters):
        randIndices = np.random.permutation(X.shape[0])
        f_tot = 0
        for batch in range(int(num_batches)):
            print 'Iteration %d batch %d of %d' % (i+1,batch+1,num_batches)
            ind = randIndices[np.mod(range(batch*batch_size,(batch+1)*batch_size),X.shape[0])]
            f,g = knca(A,X,y,P,kncas,knca0,ind=ind)
            f_tot += f
            dA = mo*dA - eta*g
            A += dA
        print 'Iteration %d complete. Objective: %s' % (i+1,f_tot/num_batches)
        obj_hist.append(f_tot/num_batches)

        if (Xtest is not None and ytest is not None):
            test_acc = ku.run_knn(Xtest,ytest,X,y,A,kmin=test_kmin,kmax=test_kmax)
            test_mat = np.vstack((np.arange(1,16,2)[None,:],test_acc[None,:]))
            test_acc_history.append(test_acc)
        print 'Iteration %d test accuracy (each column is a different test k): %s' % (i+1,test_mat)

    return A,obj_hist,test_acc_history

def knca(A,X,y,P,kncas,knca0,ind=None):
    """
    Computes the knca objective and gradient given data.
    Args:
        A: The matrix of parameters which has size #data dimensions x #projection dimensions
        X: The training data where each row corresponds to an example.
        y: A vector of labels from 0 to #classes - 1.
        P: The dimension of the projection (X.shape[1] corresponds to full-dimensional).
        kncas and knca0: object for performing inference.

    Optional Args:
        ind: If not None then these specify the rows to use in order to compute an approximate knca objective (i.e., a minibatch).

    Returns:
        The knca objective value on the data (possibly for only a subset of rows) and gradient.
    """
    N = X.shape[0]   # num points
    D = X.shape[1]   # input dimension
    C = int(max(y)+1)    # num classes

    sorted_ind = np.argsort(y)
    X = X[sorted_ind,:]
    y = y[sorted_ind]

    if (ind is None):
        ind = range(N)
    else:
        ind = sorted_ind[ind] 

    dA = np.zeros(A.shape)

    Xp = np.dot(X, A.T)   # NxP  low dim projections
    dists = cdist(Xp[ind,:], Xp, 'sqeuclidean')
    dists = np.exp(-dists)

    for (i,ii) in enumerate(ind):
        dists[i,ii] = 0

    y = y[ind].astype(np.double)

    node_marg0_all, log_Z0 = knca0.infer(dists,y)

    dthetas = np.zeros((len(ind),N))
    Z1 = 0
    for knca in kncas:
        node_marg1_all, log_Z1Kp = knca.infer(dists,y)
        if (any(np.isnan(log_Z1Kp))):
            pickle.dump({'logZ':log_Z1Kp,'dists':dists},open('diagnostics','wb'))
            raise Exception('Numerical error in knca training.')
        Z1exp = np.exp(log_Z1Kp)
        node_marg1_all[np.isnan(node_marg1_all)] = 0
        dthetas += node_marg1_all*Z1exp[:,None]
        Z1 += Z1exp

    gains = np.log(Z1) - log_Z0

    dthetas = (dthetas.T / Z1).T
    dthetas -= node_marg0_all

    dT = 0
    for (i,ii) in enumerate(ind):
        Xdiff = (X[ii,:] - X)
        dT -= np.dot(Xdiff.T,dthetas[i,:][:,None]*Xdiff)
    dA = 2*np.dot(A,dT)

    f = -np.sum(gains)/len(ind)
    g = -dA / len(ind)

    return f,g

if __name__ == "__main__":
    demo_knca(seed=1,noise=0)
