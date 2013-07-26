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

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pylab as plt
except:
    pass

import sum_cardinality as sc
import pickle
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as hac
import pylab

HUGE_VAL = 1e8

def label_selection(X, Y, label_set):
    """
assumption: X is NxD (2D), Y is N (1D)
"""

    X_ans = []
    Y_ans = []

    for l in label_set:
        I = (Y==l)
        X_ans.extend(X[I]) 
        Y_ans.extend([l]*I.sum())

    return np.array(X_ans), np.array(Y_ans)



def plot_embedding(Z,Y,K,eta,mo,obj,target_entropy=1,iters=-1,tot_iters=-1,class_to_keep=[],seed=1,dataset_name='usps',text_labels=None): 

    if dataset_name == 'usps': 
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '0.25', '0.50', '0.75']

    if iters != -1: 
        plt.figure()
        ax1 = plt.axes(frameon=False)
        ax1.axes.get_yaxis().set_visible(False)
        ax1.axes.get_xaxis().set_visible(False)

        plt.axis([-5,5, -5,5])

        print 'min/max:', np.max(Z[:,0]), np.max(Z[:,1]), np.min(Z[:,0]), np.min(Z[:,1])
        for i in xrange(Z.shape[0]):
            if text_labels is None: 
                plt.text(Z[i,0], Z[i,1], str(Y[i]), fontsize=6, color=colors[Y[i]])
            else:
                plt.text(Z[i,0], Z[i,1], text_labels[i], fontsize=6, color=colors[Y[i]])

        plt.title("USPS -- K=%s,iter=%d,eta=%.3f,mo=%.3f" % (K,iters,eta,mo))
        if len(class_to_keep) == 0: 
            fig_filename = 'figs/%s_ktsne_K=%d_eta=%.3f_mo=%.3f_targH=%.2f_iter=%d_seed=%d' % (dataset_name,K,eta,mo,target_entropy,iters,seed)
        else:
            fig_filename = 'figs/%s_ktsne_K=%d_eta=%.3f_mo=%.3f_targH=%.2f_ctk=%s_iter=%d_seed=%d' % (dataset_name,K,eta,mo,target_entropy,''.join(map(str,class_to_keep)),iters,seed)

        plt.savefig(fig_filename + '.png' )
        plt.savefig(fig_filename + '.pdf' )
        plt.savefig(fig_filename + '.eps' )

        if iter==1 or iters%50 == 0: 
           pickle_file = open('res/%s_z_K=%d_eta=%.3f_mo=%.2f_targH=%.2f_iter=%d_seed=%d.pkl' % (dataset_name,K,eta,mo,target_entropy,iters,seed), 'w') 
           pickle.dump(Z, pickle_file)
           pickle.dump(Y, pickle_file)
           pickle.dump(obj, pickle_file)
           pickle_file.close()


def plot_knca(X,Y,A):
    s = ['ob','rs','gd']
    j = 0
    Xp = np.dot(X,A.T)
    pylab.hold('on')
    for i in np.unique(Y):
        pylab.plot(Xp[Y==i,0],Xp[Y==i,1],s[j])
        j += 1
    pylab.hold('off')

def printf(x):
    from sys import stdout
    stdout.write(x)
    stdout.flush()

def dist_normalize(dijs, Y, perplexity=1,dataset_name='usps'):
    K=perplexity

    sigmas = np.zeros(dijs.shape[0]); 
    dijs_scaled = dijs; 
    N = dijs.shape[0]
    for i in xrange(N):
        if dataset_name == 'usps':
            sigma_low = 1e1; 
            sigma_high = 1e6; 
        di_ = dijs[i,:]
        
        it=0
        while True: # binary search 
            sigma = (sigma_high - sigma_low)/2. + sigma_low

            di = di_/(2*sigma**2) 
            #pi = softmax(di) 
            di[i] = 0 #-1e10

            
            #entropy = -(pi * np.log(pi)).sum() 
            entropy = H(di)
            if not entropy >=0 :
                import pdb
                pdb.set_trace()

            assert entropy>=0, 'entropy is negative'
                
            if abs(entropy-np.log(perplexity)) < 1:  # between -1 and 1 
                try:
                    printf('%d/%d: (it=%d, entropy=%.3f (target=%.3f), sigma_low=%.2f, sigma=%.2f, sigma_high=%.2f, label=%d)\r' % (i, dijs.shape[0], it, entropy, np.log(perplexity), sigma_low, sigma, sigma_high, Y[i])) 
                except IndexError:
                    import pdb
                    pdb.set_trace()
                sigmas[i] = sigma;
                dijs_scaled[i,:] /= (2*sigma**2); 
                break 
            elif (entropy-np.log(perplexity)) > 1: # > 1
                sigma_high = sigma
            else: # < 1
                sigma_low = sigma

            it+=1

    print 'dist normalize complete (N=%s)' % N
    return dijs_scaled


def softmax(a):
    a=a-a.max() #1).reshape(-1,1)
    b=np.exp(a)
    c=b.sum() #1)
    if np.isnan(a).any():
        import pdb
        pdb.set_trace()
    ans = b/c #.reshape(-1,1)
    if np.isnan(ans).any():
        import pdb
        pdb.set_trace()
    return ans

def H(a):
    m = a.max()
    b = a-m
    up = np.exp(b)
    p = up/up.sum()
    logZ = np.log(up.sum()) + m
    return -(p*(a-logZ)).sum()

def train_sgd(X,Y,dataset_name,P,K,Z=None,num_iters=1,perplexity=5,batch_size=10,eta=0.1,mo=0,L2=0,class_to_keep=[],seed=1,distance='sqeuclidean',text_labels=None):
    """
X, Y: points and their respective labels 
dataset_name: the name given to this dataset
P: the dimensionality of the embedding space
K: the number of neighbors 
Z: coordinates of the points in embedded space
num_iters: number of passes over the set of points  
perplexity: perplexity of the distribution over datapoints. Check: 
            'Visualizing Data using t-SNE by van der Maaten and Hinton in JMLR (09) 2008', 
            for details. 
batch_size: size of mini batches
eta: learning rate
mo: momentum
L2: size of L2 regularizer
class_to_keep: subset of labels that have been kept (for naming results and figure files)
seed: the value of the seed (for naming results and figure files)
distance: distance metric {sqeuclidean, emd}
test_labels: alternative labels used in plots
    """

    N = X.shape[0]   # num points
    D = X.shape[1]   # input dimension

    ##################################################################################
    if Z is None:  Z = np.random.randn(N,P)*0.1  # points in embedded space

    if P == 2: # plot if embedding is in 2D
        plot_embedding(Z,Y,K,eta,mo,0,target_entropy=perplexity,iters=0,tot_iters=num_iters,class_to_keep=class_to_keep,seed=seed,dataset_name=dataset_name,text_labels=text_labels)

    if distance == 'sqeuclidean': 
        dijs = -squareform(pdist(X, 'sqeuclidean'))
    elif distance == 'emd': # earth mover's distance
        from emd.emddist import emddist
        dijs = -emddist(X)
    dijs = dist_normalize(dijs,Y,perplexity=perplexity,dataset_name=dataset_name) 
    dijs -= HUGE_VAL * np.eye(N)
    pijs = np.zeros((N,N))

    T = sc.make_balanced_binary_tree(N)
    root_idx = N + T.shape[0]-1
    print "Root idx", root_idx

    global_count_potential = np.zeros(N+1)
    global_count_potential[K] = 1
    only_global_constraint_dict = {}
    only_global_constraint_dict[root_idx] = global_count_potential

    # Precompute E_i[y_j] (or pijs)
    print 'precomute E_i[y_j]'

    from chain.chain_alg_wrapper import ChainAlg
    chain = ChainAlg(N, Kmin=K, Kmax=K, minibatch_size = N)
    pot = dijs
    #exp_pot = np.exp(dijs)
    marginals, samples, logZs = chain.infer(pot)
    pijs = marginals


    debug = False
    if debug:
        pijs_debug = pijs.copy()
        for i in xrange(N):
            thetas = dijs[i,:]
            #print np.exp(thetas)
            node_margs, count_margs, log_Z = \
                sc.conv_tree(np.exp(thetas), only_global_constraint_dict, T, use_fft=False)
            pijs_debug[i,:] = node_margs

        diff = abs(pijs - pijs_debug).max() 
        assert diff < 1e-8
        print 'precomputation of marginals is correct!'


    ##################################################################################

    num_batches = np.ceil(np.double(N)/batch_size)
    randIndices = np.random.permutation(X.shape[0])

    print 'num_batches = %s'  % num_batches

    dZ = np.zeros(Z.shape)
    V = dZ*0
    for i in range(num_iters):
        f_tot = 0
        for batch in range(int(num_batches)):
            print "iteration " + str(i+1) + " batch " + str(batch+1) + " of " + str(int(num_batches))
            ind = randIndices[np.mod(range(batch*batch_size,(batch+1)*batch_size),X.shape[0])]
            f_tot += ksne_obj(Z.flatten(),X,K,P,pijs,only_global_constraint_dict,T)


            g = ksne_grad((Z+V*mo).flatten(),X,K,P,pijs,only_global_constraint_dict,T).reshape(Z.shape)

            #dZ = mo*dZ - eta*g
            V = V*mo + eta*(g-L2*(Z+V*mo))
            #Z -= dZ
            Z += V
        print 'objective: %s, |g|=%s' % (str(f_tot/num_batches), abs(g).mean())
        if P == 2: # plot if embedding is in 2D
            plot_embedding(Z,Y,K,eta,mo,f_tot/num_batches,target_entropy=perplexity,iters=i+1,tot_iters=num_iters,class_to_keep=class_to_keep,seed=seed,dataset_name=dataset_name,text_labels=text_labels)

    return Z

def knn(Xtest,Ytest,Xtrain,Ytrain,A,K):
    C = int(Ytrain.max()+1)
    Xtestp = np.dot(Xtest,A.T)
    Xtrainp = np.dot(Xtrain,A.T)

    num_errs = 0
    for i in range(Xtest.shape[0]):
        dists = -np.sum((Xtestp[i,:][None,:] - Xtrainp)**2,1)
        dists[i] = -np.inf
        cutoffs = np.argsort(dists)[-K:]
        class_votes = np.zeros(C)
        for k in range(C):
            class_votes[k] += np.sum(Ytrain[cutoffs]==k)
        num_errs += class_votes.argmax() != Ytest[i]

    return num_errs / np.double(Ytest.shape[0])

def train_ksne_lbfgs(X,Y,P,K,mf=10):
    N = X.shape[0]   # num points
    D = X.shape[1]   # input dimension

    Z = np.random.randn(N,P)  # points in embedded space

    T = sc.make_balanced_binary_tree(N)
    root_idx = N + T.shape[0]-1

    global_count_potential = np.zeros(N+1)
    global_count_potential[K] = 1
    only_global_constraint_dict = {}
    only_global_constraint_dict[root_idx] = global_count_potential

    res = lbfgs(sne_obj, A.flatten(), fprime=sne_grad, args=(X,Y,K,P,T,only_global_constraint_dict,roots), maxfun=mf, disp=1)

    return res[0].reshape(A.shape)


def ksne_obj(params,X,K,P,pijs,count_pot_dict,T, fast=True):
    N = X.shape[0]
    Z = params.reshape((N, P))
    tdijs = squareform(pdist(Z, 'sqeuclidean'))  # tilde dijs
    from numpy import log
    tdijs = -log(1+tdijs)

    term1 = np.sum(tdijs * pijs)  # diagonal is 0

    if fast:
        from chain.chain_alg_wrapper import ChainAlg
        chain = ChainAlg(N, Kmin=K, Kmax=K, minibatch_size = N)

        mask = np.ones((N,N)) - np.diag(np.ones(N))
        log_mask = -np.diag(np.ones(N))*1e100



        pot = tdijs + log_mask
        #exp_pot = mask * np.exp(tdijs)
        marginals, samples, logZs = chain.infer(pot)
        sum_log_Z = logZs.sum()

    else:
        sum_log_Z = 0
        mask = np.ones(N)
        for i in xrange(N):
            if i > 0:  mask[i-1] = 1
            mask[i] = 0
    
            node_margs, count_margs, log_Z = \
                sc.conv_tree(mask * np.exp(tdijs[i,:]), count_pot_dict, T, use_fft=False)
            sum_log_Z += log_Z
    
    return term1 - sum_log_Z


def ksne_grad(params,X,K,P,pijs,count_pot_dict,T,chain=None):

    N = len(X)
    Z = params.reshape((N, P))
    tdijs0 = squareform(pdist(Z, 'sqeuclidean'))  # tilde dijs
    from numpy import log
    tdijs = -log(1+tdijs0)

    grad2 = pijs.copy()

    if chain is None:
        from chain.chain_alg_wrapper import ChainAlg
        chain = ChainAlg(N, Kmin=K, Kmax=K, minibatch_size = N)

    mask = np.ones((N,N)) - np.diag(np.ones(N))
    log_mask = -np.diag(np.ones(N))*1e100

    pot = tdijs + log_mask
    exp_pot = mask * np.exp(tdijs)
    marginals, samples, logZs = chain.infer(pot)
    grad2 -= marginals 
    grad = grad2 / (1 + tdijs0)


    g2 = 2*(grad+grad.T)
    dZ = g2.dot(Z)-g2.sum(1)[:,np.newaxis]*Z

    return dZ


def ksne_grad_correct(params,X,K,P,pijs,count_pot_dict,T):
    Z = params.reshape((N, P))
    tdijs = -squareform(pdist(Z, 'sqeuclidean'))  # tilde dijs

    grad = pijs.copy()

    mask = np.ones(N)
    for i in xrange(N):
        if i > 0:  mask[i-1] = 1
        mask[i] = 0

        node_margs, count_margs, log_Z = \
            sc.conv_tree(mask * np.exp(tdijs[i,:]), count_pot_dict, T, use_fft=False)

        grad[i,:] -= node_margs

    g2 = 2*(grad+grad.T)
    dZ = g2.dot(Z)-g2.sum(1)[:,np.newaxis]*Z

    return dZ


def check_grad(params,step,X,K,P,pijs,count_pot_dict,T):
    inds = np.array([0])
    g_calc = ksne_grad(params,X,K,P,pijs,count_pot_dict,T)
    g_num = np.zeros(params.shape[0])
    fA = ksne_obj(params,X,K,P,pijs,count_pot_dict,T, fast=False)
    fB = ksne_obj(params,X,K,P,pijs,count_pot_dict,T, fast=True)
    print 'check_grad: slow obj = %s' % fA
    print 'check_grad: fast obj = %s' % fB
    assert abs(fA-fB)<1e-6
    print 'success: fast objective = slow objective'

    for i in range(params.shape[0]):
        params[i] += step
        f1 = ksne_obj(params,X,K,P,pijs,count_pot_dict,T)
        params[i] -= 2*step
        f2 = ksne_obj(params,X,K,P,pijs,count_pot_dict,T)
        params[i] += step

        g_num[i] = (f1-f2)/(2*step)
    return g_calc, g_num


if __name__ == "__main__":
    import sys

    P = 2   # embedding dimension

    if len(sys.argv) < 9: 
        print 'usage: %s datasetname k num_iters perplexity L2 learning_rate number_of_examples_to_use seed' % sys.argv[0]
        sys.exit(0)

    #dataset_name, k, num_iters, perplexity, L2, eta, num_cases, seed = sys.argv[1:]
    dataset_name= sys.argv[1]
    k          = int(sys.argv[2])
    num_iters  = int(sys.argv[3])
    perplexity = int(sys.argv[4])
    L2         = float(sys.argv[5])
    eta        = float(sys.argv[6])
    num_cases  = int(sys.argv[7])
    seed       = int(sys.argv[8])

    print 'PARAMS:'
    print '\tdataset_name = %s' % dataset_name
    print '\tk = %d' % k 
    print '\tnum_iters = %s' % num_iters
    print '\tperplexity = %s' % perplexity
    print '\tL2 = %s' % L2
    print '\teta = %s' % eta
    print '\tnum_cases = %d' % num_cases
    print '\tseed = %d' % seed


    if dataset_name == 'usps':

        # python learn_ktsne.py 1 250 2 0.5 0.1 1000 1
        from usps import get_usps_split
        i=0

        (X_train, Y_train), (X_test, Y_test) = get_usps_split(1) 

        np.random.seed(seed)

        X_train = X_train[:num_cases]
        Y_train = Y_train[:num_cases] 

        [N, D] = X_train.shape
        print 'USPS, num_examples=%d, num_dim=%d' % (N, D) 

        Z = train_sgd(X_train,Y_train,dataset_name,P,k, num_iters=num_iters, perplexity=perplexity, eta=eta,mo=.9,L2=L2, batch_size=N,seed=seed)

        plt.clf()
        plt.hold(True)
        syms = ['bo', 'ro', 'go', 'bx', 'rx', 'gx', 'b<', 'r<', 'g<', 'k.']
        assert len(syms)==10
        for i,sym in zip(range(10),syms):
            I = (Y_train==i)
            plt.plot(Z[I,0], Z[I,1], sym)

        plt.title("USPS -- K=%s" % k)
        fig_filename = 'usps_K=%d' % (k)
        plt.savefig(fig_filename + '.png' )
        plt.savefig(fig_filename + '.pdf' )
        plt.savefig(fig_filename + '.eps' )
        plt.show()
