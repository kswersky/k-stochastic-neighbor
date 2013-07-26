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
import sys
import scipy.signal as sig
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import pdist, squareform
from time import clock, time
try:
    import matplotlib
    import matplotlib.pylab as plt
except:
    pass

FFT_CROSSOVER_POINT = 32

def neg_energy(assn, node_potentials, count_potential):
    nen = np.dot(assn, node_potentials) + count_potential[np.sum(assn)]
    return nen


def idx_to_assn(idx, D):
    """ convert 1D index to setting of binary variables """
    return np.array([(idx / 2**d) % 2 for d in range(D)])


def random_categorical(probs):
    return probs.cumsum().searchsorted(np.random.rand())


def hierarchical_cluster(trainx):
    """ See the scipy.cluster.hierarchy documentation for the
    meanings of entries in T.

    The result can be plotted by calling hac.dendrogram(T). """    

    T = hac.complete(pdist(trainx.T) + .1)
    return T


def make_balanced_binary_tree(D):
    """ Make a tree structure to be used by the convolution tree
    algorithm, where the tree is as balanced as possible.

    The result can be plotted by calling hac.dendrogram(T). """

    if D == 1:  return np.zeros((0,4))

    T = np.zeros((D-1,4))
    ctr = 0
    top_level = np.arange(D)
    new_top_level = np.zeros(np.ceil(D/2.))
    lvl = 1
    while new_top_level.shape[0] > 1:
        for d in range(top_level.shape[0]/2):
            new_top_level[d] = ctr+D

            T[ctr,0] = top_level[2*d]
            T[ctr,1] = top_level[2*d+1]
            T[ctr,2] = lvl
            T[ctr,3] = 0
            if T[ctr,0] < D:  T[ctr,3] += 1
            else:             T[ctr,3] += T[T[ctr,0]-D,3]
            if T[ctr,1] < D:  T[ctr,3] += 1
            else:             T[ctr,3] += T[T[ctr,1]-D,3]
                
            ctr += 1

        if top_level.shape[0] % 2 == 1: new_top_level[-1] = top_level[-1]

        top_level = new_top_level
        new_top_level = np.zeros(np.ceil(top_level.shape[0]/2.))
        lvl += 1

    T[ctr,0] = top_level[0]
    T[ctr,1] = top_level[1]
    T[ctr,2] = lvl
    if T[ctr,0] < D:  T[ctr,3] += 1
    else:             T[ctr,3] += T[T[ctr,0]-D,3]
    if T[ctr,1] < D:  T[ctr,3] += 1
    else:             T[ctr,3] += T[T[ctr,1]-D,3]
            
    return T


def runtime_test(Ds):
    results = np.zeros((len(Ds), 4))
    for d, D in enumerate(Ds):
        print D
        tree, fft, chain = test_our_algs(D, 1)
        results[d,0] = D
        results[d,1:] = [tree, fft, chain]

    return results

def plot_runtime_results(results):
    plt.rcParams["figure.figsize"] = 7,7
    plt.rcParams["font.size"] = 22
    matplotlib.rc("xtick", labelsize=24)
    matplotlib.rc("ytick", labelsize=24)

    params = {"text.fontsize" : 32,
              "font.size" : 32,
              "legend.fontsize" : 30,
              "axes.labelsize" : 32,
              "text.usetex" : False
              }
    plt.rcParams.update(params)
    
    #plt.semilogx(results[:,0], results[:,3], 'r-x', lw=3)
    #plt.semilogx(results[:,0], results[:,1], 'g-D', lw=3)
    #plt.semilogx(results[:,0], results[:,2], 'b-s', lw=3)

    plt.plot(results[:,0], results[:,3], 'r-x', lw=3, ms=10)
    plt.plot(results[:,0], results[:,1], 'g-D', lw=3, ms=10)
    plt.plot(results[:,0], results[:,2], 'b-s', lw=3, ms=10)

    plt.legend(["Chain", "Tree", "FFT Tree"], loc="upper left")
    plt.xticks([1e5, 2e5, 3e5])
    plt.yticks([0, 60, 120, 180])

    plt.xlabel("Problem Size")
    plt.ylabel("Runtime (sec)")
    return results


def test_our_algs(D, num_runs, count_cap=None):
    from time import time, clock

    # need a tree structure for the convolution tree algorithm.
    T = make_balanced_binary_tree(D)

    tree_time = 0
    fftree_time = 0
    chain_time = 0
    for r in range(num_runs):
        exp_node_pots = np.exp(1 * np.random.randn(D))
        exp_count_pots = np.exp(0 * np.random.randn(D+1))
        if count_cap is not None:
            exp_count_pots[count_cap:] = -np.inf            

        exp_count_pot_dict = {}  # dictionary mapping internal node indexes to exp count potentials
        root_idx = D + T.shape[0]-1
        exp_count_pot_dict[root_idx] = exp_count_pots

        start = clock()
        nm_conv, cm_conv, Z_conv = conv_tree(exp_node_pots, exp_count_pot_dict, T,
                                             use_fft=False)
        cm_conv = cm_conv[root_idx]
        tree_time += (clock() - start)
            
        start = clock()
        nm_fconv, cm_fconv, Z_fconv = conv_tree(exp_node_pots, exp_count_pot_dict, T,
                                                use_fft=True)
        cm_fconv = cm_fconv[root_idx]
        fftree_time += (clock() - start)

        if D <= 20000:
            start = clock()
            nm_chain, cm_chain = pass_all_messages(exp_node_pots, exp_count_pots)
            chain_time += (clock() - start)

            # Make sure marginals agree to within eps
            if False:
                print cm_chain, cm_conv, cm_fconv
                assert np.sum((nm_conv-nm_chain)**2) < 1e-8
                assert np.sum((cm_conv-cm_chain)**2) < 1e-8
                
                assert np.sum((nm_fconv-nm_chain)**2) < 1e-8
                assert np.sum((cm_fconv-cm_chain)**2) < 1e-8
        else:
            chain_time = 0

    avg_tree_time = tree_time / num_runs
    avg_fftree_time = fftree_time / num_runs
    avg_chain_time = chain_time / num_runs

    print "  Tree time \t%s" % avg_tree_time
    print "FFTree time \t%s" % avg_fftree_time
    print " Chain time \t%s" % avg_chain_time

    return avg_tree_time, avg_fftree_time, avg_chain_time


def test_convolution_speeds(D, mode="full"):
    """ fftconvolve should be *much* faster than the others.
    if it's not (as it wasn't for me originally), you probably
    need to upgrade your scipy version -- there was a bug in
    previous versions that caused it to be very slow for
    some inputs. """

    from time import time, clock

    a = np.random.rand(D)
    b = np.random.rand(D)

    functions = [np.convolve, sig.convolve, sig.fftconvolve]

    times = np.zeros(len(functions))
    for f, fn in enumerate(functions):
        start = clock()
        c = fn(a,b,mode=mode)
        times[f] += (clock() - start)    
    return times


def conv_tree(exp_node_pots, exp_count_pot_dict, T,
              use_fft=True, VERBOSE=False, count_cap=None):

    if count_cap is None:
        D = exp_node_pots.shape[0]
        root_idx = D + T.shape[0]-1
        #count_cap = np.max(np.nonzero(exp_count_pot_dict[root_idx] > 0))
        count_cap = D
        #if True:  print "Setting count_cap = ", count_cap

    # easier to special case D=1 than make code below work for it
    if exp_node_pots.shape[0] == 1:
        p0 = 1. / (1. + exp_node_pots[0])
        p1 = exp_node_pots[0] / (1. + exp_node_pots[0])        

        if 1 in exp_count_pot_dict:
            p0 *= exp_count_pot_dict[1][0]
            p1 *= exp_count_pot_dict[1][1]

        node_margs = np.array([p1])
        count_margs = {}
        count_margs[0] = np.array([p0, p1])
        log_Z = np.log(1. + exp_node_pots[0])
        return node_margs, count_margs, log_Z        

    if False and count_cap is not None:
        for node in exp_count_pot_dict:
            exp_count_pot_dict[node] = exp_count_pot_dict[node][:count_cap]

    TIME_up = 0
    #TIME_uconv = 0
    TIME_down = 0
    #TIME_dconv = 0

    D = exp_node_pots.shape[0]
    cards = np.zeros(2*D-1, dtype=np.int)
    cards[:D] = 1

    if False:  # slow way
        # traverse the tree upwards to compute node cardinalities -- slow
        for merge in range(T.shape[0]):
            cards[D+merge] = cards[T[merge,0]] + cards[T[merge,1]]
            cards[D+merge] = np.minimum(count_cap, cards[D+merge])
        cards = np.int32(cards)
    else:  # fast way
        cards[D:] = np.minimum(count_cap, T[:,3].astype(np.int))
    
    # all messages will be stored in a single array.  this
    # array lets us know where to find them.
    start_idxs = np.cumsum(np.hstack([0, cards + 1])).astype(np.int)
    up_messages   = np.zeros(np.sum(cards + 1))
    down_messages = np.zeros(np.sum(cards + 1))

    if VERBOSE:  print "allocating message arrays of size", up_messages.shape[0]

    # fill in unary potentials at leaves
    if False:  # slow way
        for d in range(D):
            start = start_idxs[d]
            end = start + cards[d] + 1
            if np.isinf(exp_node_pots[d]):
                up_messages[start:end] = [0, 1]
            else:
                up_messages[start:end] = [1, exp_node_pots[d]]
    else:  # fast way
        up_messages[:2*D:2] = 1
        up_messages[1:2*D:2] = exp_node_pots

    if VERBOSE:
        print cards.shape, D, T.shape
        print "Cards, starts"
        print cards
        print start_idxs
        print "Initial msgs"
        print up_messages

    log_Z = 0
    # Upward pass
    start_up = time()
    for m in range(T.shape[0]):
        # merging T[m,0] and T[m,1] to get node dd
        dd = D + m  # index of parent node
        ch1, ch2 = int(T[m,0]), int(T[m,1]) # indices of children nodes
        
        start_ch1 = start_idxs[ch1];   end_ch1 = start_ch1 + cards[ch1] + 1
        start_ch2 = start_idxs[ch2];   end_ch2 = start_ch2 + cards[ch2] + 1
        start_p   = start_idxs[dd];    end_p   = start_p   + cards[dd]  + 1

        use_fft_here = use_fft and np.minimum(cards[ch1], cards[ch2]) > FFT_CROSSOVER_POINT
        
        ch1_msg = up_messages[start_ch1:end_ch1]
        ch2_msg = up_messages[start_ch2:end_ch2]

        # multiply in any subset count potentials
        if ch1 in exp_count_pot_dict:  ch1_msg *= exp_count_pot_dict[ch1]
        if ch2 in exp_count_pot_dict:  ch2_msg *= exp_count_pot_dict[ch2]

        #start_conv = time()
        if use_fft_here:
            up_messages[start_p:end_p] = sig.fftconvolve(ch1_msg, ch2_msg, mode="full")[:count_cap+1]
        else:
            up_messages[start_p:end_p] = np.convolve(ch1_msg, ch2_msg, mode="full")[:count_cap+1]

        #TIME_uconv += (time() - start_conv)
        
        # normalize messages for numerical reasons, but store constants
        # so we can compute the partition function (Z)
        Z_m = np.sum(up_messages[start_p:end_p])
        assert Z_m != 0, "Partition function is 0!"

        up_messages[start_p:end_p] /= Z_m
        
        log_Z += np.log(Z_m)

    TIME_up += (time() - start_up)

    # Last contribution to the partition function
    count_beliefs = up_messages[start_p:end_p].copy()
    root_idx = D + T.shape[0]-1
    if root_idx in exp_count_pot_dict:
        count_beliefs *= exp_count_pot_dict[D + T.shape[0]-1][:count_cap+1]
    Z_ct = np.sum(count_beliefs)

    assert Z_ct != 0, "Partition function is 0!"

    count_beliefs /= Z_ct
    
    if Z_ct < 1e-10:
        #print "Warning Z_ct=%s (size=%s, ct[0]=%s)" %(Z_ct, D, exp_count_pot_dict[D+T.shape[0]-1][0])
        #print np.prod(1. / (1.0 + exp_node_pots))
        pass

    log_Z += np.log(Z_ct)
    
    if False:
        print "End upward pass up messages"
        print start_p, end_p, up_messages[start_p:end_p]

        print "count beliefs"
        print count_beliefs
        print

    # Downward pass
    # set count potential to be uniform at root.  will get multiplied after down_p_msg
    # is created from root to children
    start_down = time()
    down_messages[start_p:end_p] = 1 #np.ones(count_cap+1)
    count_margs = {}
    for m in reversed(range(T.shape[0])):
        # merged T[m,0] and T[m,1] to get node dd.
        # now need to send message from parent down to children
        dd = D + m  # index of parent node
        ch1, ch2 = int(T[m,0]), int(T[m,1])  # indices of children nodes    

        start_ch1 = start_idxs[ch1];   end_ch1 = start_ch1 + cards[ch1] + 1
        start_ch2 = start_idxs[ch2];   end_ch2 = start_ch2 + cards[ch2] + 1
        start_p   = start_idxs[dd];    end_p   = start_p   + cards[dd]  + 1

        # Add just enough padding of 0's so that convolutions can be
        # done with a simple call to convolve with mode="valid"
        down_p_msg = np.zeros(cards[ch1] + cards[ch2] + 1)
        down_p_msg[:end_p-start_p] = down_messages[start_p:end_p]
        if dd in exp_count_pot_dict:
            #print dd, "down_p before", down_p_msg
            down_p_msg *= exp_count_pot_dict[dd][:down_p_msg.shape[0]]
            #print dd, "down_p  after", down_p_msg

            count_margs[dd] = down_p_msg[:cards[dd]+1] * up_messages[start_p:end_p]
            count_margs[dd] /= np.sum(count_margs[dd])

        # could reverse child messages like this...
        ch1_msg = up_messages[start_ch1:end_ch1]  
        ch2_msg = up_messages[start_ch2:end_ch2]
        ch1_rev_msg = ch1_msg[::-1] # reversed!
        ch2_rev_msg = ch2_msg[::-1] # reversed!

        # ... but it's faster to do it all in one go
        #if start_ch1 == 0:  ch1_rev_msg = up_messages[end_ch1::-1]
        #else:               ch1_rev_msg = up_messages[end_ch1:start_ch1-1:-1]
        #if start_ch2 == 0:  ch2_rev_msg = up_messages[end_ch2::-1]
        #else:               ch2_rev_msg = up_messages[end_ch2:start_ch2-1:-1]
            
        use_fft_here = use_fft and np.minimum(cards[ch1], cards[ch2]) > FFT_CROSSOVER_POINT

        #start_conv = time()
        if use_fft_here:
            down1 = sig.fftconvolve(down_p_msg, ch2_rev_msg, mode="valid")            
            down2 = sig.fftconvolve(down_p_msg, ch1_rev_msg, mode="valid")
            down1 = np.maximum(down1, 0)
            down2 = np.maximum(down2, 0)
        else:
            down1 = np.convolve(down_p_msg, ch2_rev_msg, mode="valid")
            down2 = np.convolve(down_p_msg, ch1_rev_msg, mode="valid")

        #TIME_dconv += (time() - start_conv)

        if False:
            full_down = np.zeros(T[m,3]+1)
            full_down[:down_p_msg.shape[0]] = down_p_msg

            if ch1 < D:  real_ch1_card = 1
            else:        real_ch1_card = T[ch1-D,3]
            if ch2 < D:  real_ch2_card = 1
            else:        real_ch2_card = T[ch2-D,3]

            print T[m,3], real_ch1_card, real_ch2_card
            print cards[dd], cards[ch1], cards[ch2]
            print down_p_msg.shape[0], ch1_rev_msg.shape[0], ch2_rev_msg.shape[0]
                
            full_ch1 = np.zeros(real_ch1_card+1)
            full_ch1[:ch1_rev_msg.shape[0]] = ch1_rev_msg[::-1]
            full_ch2 = np.zeros(real_ch2_card+1)
            full_ch2[:ch2_rev_msg.shape[0]] = ch2_rev_msg[::-1]

            full_ch1 = full_ch1[::-1]
            full_ch2 = full_ch2[::-1]
            
            print "Fulls"
            print full_down
            print full_ch1
            print full_ch2
                      
        # Pull out the right part of down1/down2 to use as the downward message.  
        cstart_ch1 = down1.shape[0] - cards[ch1]-1
        cend_ch1 = down1.shape[0]
        cstart_ch2 = down2.shape[0] - cards[ch2]-1
        cend_ch2 = down2.shape[0]

        #print dd, "d1", down1, down1[cstart_ch1:cend_ch1], start_ch1, end_ch1
        #print dd, "d2", down2, down2[cstart_ch2:cend_ch2], start_ch2, end_ch2
        
        down_messages[start_ch1:end_ch1] = down1[cstart_ch1:cend_ch1]
        down_messages[start_ch2:end_ch2] = down2[cstart_ch2:cend_ch2]

        # to combat over/under-flow -- could be less aggressive here to make it faster
        Z1 = np.sum(down_messages[start_ch1:end_ch1])
        Z2 = np.sum(down_messages[start_ch2:end_ch2])
        down_messages[start_ch1:end_ch1] /= Z1
        down_messages[start_ch2:end_ch2] /= Z2
            
    TIME_down += (time() - start_down)
    
    node_beliefs = down_messages[:(2*D)] * up_messages[:(2*D)]

    b0 = node_beliefs[::2]
    b1 = node_beliefs[1::2]
    node_margs = b1 / (b0 + b1)
    
    if VERBOSE:
        #print "Node marginals"
        #print node_margs

        print "Times"
        print "   UP", TIME_up
        #print "uCONV", TIME_uconv
        print " DOWN", TIME_down
        #print "dCONV", TIME_dconv

    return node_margs, count_margs, log_Z



def up_conv_tree_down_sample(exp_node_pots, exp_count_pot_dict, T,
                             use_fft=True, VERBOSE=False):

    D = exp_node_pots.shape[0]
    cards = np.zeros(2*D-1)
    cards[:D] = 1

    # traverse the tree upwards to compute node cardinalities
    for merge in range(T.shape[0]):
        cards[D+merge] = cards[T[merge,0]] + cards[T[merge,1]]
    cards = np.int32(cards)

    # all messages will be stored in a single array.  this
    # array lets us know where to find them.
    start_idxs = np.cumsum(np.hstack([0, cards + 1]))
    up_messages   = np.zeros(np.sum(cards + 1))
    down_messages = np.zeros(np.sum(cards + 1))

    # fill in unary potentials at leaves
    for d in range(D):
        start = start_idxs[d]
        end = start + cards[d] + 1
        if np.isinf(exp_node_pots[d]):
            up_messages[start:end] = [0, 1]
        else:
            up_messages[start:end] = [1, exp_node_pots[d]]


    # Upward pass -- copied from conv_tree
    #start_up = time()
    for m in range(T.shape[0]):
        # merging T[m,0] and T[m,1] to get node dd
        dd = D + m  # index of parent node
        ch1, ch2 = int(T[m,0]), int(T[m,1]) # indices of children nodes
        
        start_ch1 = start_idxs[ch1];   end_ch1 = start_ch1 + cards[ch1] + 1
        start_ch2 = start_idxs[ch2];   end_ch2 = start_ch2 + cards[ch2] + 1
        start_p   = start_idxs[dd];    end_p   = start_p   + cards[dd]  + 1

        ch1_msg = up_messages[start_ch1:end_ch1]
        ch2_msg = up_messages[start_ch2:end_ch2]

        # multiply in any subset count potentials
        if ch1 in exp_count_pot_dict:  ch1_msg *= exp_count_pot_dict[ch1]
        if ch2 in exp_count_pot_dict:  ch2_msg *= exp_count_pot_dict[ch2]

        use_fft_here = use_fft and np.minimum(cards[ch1], cards[ch2]) > FFT_CROSSOVER_POINT
        if use_fft_here:
            up_messages[start_p:end_p] = sig.fftconvolve(ch1_msg, ch2_msg, mode="full")
        else:
            up_messages[start_p:end_p] = np.convolve(ch1_msg, ch2_msg, mode="full")
        
        # normalize messages for numerical reasons, but store constants
        # so we can compute the partition function (Z)
        Z_m = np.sum(up_messages[start_p:end_p])
        assert Z_m != 0, "Partition function is 0!"
        up_messages[start_p:end_p] /= Z_m

    # Downward sampling pass
    count_margs = {}
    num_on = {}
    root_idx = D + T.shape[0]-1
    root_start = start_idxs[root_idx];  root_end = root_start + cards[root_idx] + 1
    root_count_dist = up_messages[root_start:root_end]
    if root_idx in exp_count_pot_dict:  root_count_dist *= exp_count_pot_dict[root_idx]
    root_count_dist /= np.sum(root_count_dist)

    num_on[root_idx] = random_categorical(root_count_dist)
    #print "root: %s on" % num_on[root_idx]

    for m in reversed(range(T.shape[0])):
        # merged T[m,0] and T[m,1] to get node dd.
        # now need to send message from parent down to children
        dd = D + m  # index of parent node
        ch1, ch2 = int(T[m,0]), int(T[m,1])  # indices of children nodes    

        start_ch1 = start_idxs[ch1]
        start_ch2 = start_idxs[ch2]

        parent_ct = num_on[dd]
        # now need to sample children values given that their sum is p_ct
        ch1_dist = np.zeros(cards[ch1]+1)
        for ch1_ct in range(cards[ch1]+1):
            ch2_ct = parent_ct - ch1_ct
            #print "cts", parent_ct, "%s/%s" % (ch1_ct, cards[ch1]), "%s/%s" % (ch2_ct, cards[ch2])

            if ch2_ct < 0 or ch2_ct > cards[ch2]:  continue   # assign this 0 prob

            ch1_dist[ch1_ct] = up_messages[start_ch1+ch1_ct] * up_messages[start_ch2+ch2_ct]
            
            if ch1 in exp_count_pot_dict:  ch1_dist[ch1_ct] *= exp_count_pot_dict[ch1][ch1_ct]
            if ch2 in exp_count_pot_dict:  ch1_dist[ch1_ct] *= exp_count_pot_dict[ch2][ch2_ct]

        #print ch1_dist
        ch1_dist /= np.sum(ch1_dist)

        num_on[ch1] = random_categorical(ch1_dist)
        num_on[ch2] = parent_ct - num_on[ch1]

        #print "(parent %s: %s)" % (dd, parent_ct)
        #print "node %s <-- %s" % (ch1, num_on[ch1])
        #print "node %s <-- %s" % (ch2, num_on[ch2])

    return np.array([num_on[d] for d in range(D)])


def avg_logprob_from_empirical_margs(node_potentials, count_pot_dict,
                                     empirical_margs, empirical_count_dict, log_Z):
    logprob  = np.sum(node_potentials * empirical_margs)
    for tt in count_pot_dict:
        logprob += np.sum(count_pot_dict[tt] * empirical_count_dict[tt])
    logprob -= log_Z

    return logprob


def parent_to_child_message(d, u, method="brute_force"):
    """ A little test showing how to compute downward messages
    using convolutions. """
    
    # parent = ch1 + ch2, so ch1 = parent - ch2.
    # this leads to messages to ch1 of the form:
    #
    # m(k) = \sum_i d(k+i)u(i)   for i=0 to |u|
    
    if method == "brute_force":
        D_ch2 = u.shape[0]
        D_ch1 = d.shape[0]-D_ch2
        m = np.zeros(D_ch1+1)
        for k in range(D_ch1+1):
            for i in range(D_ch2):
                m[k] += d[k+i]*u[i]
        return m
    else:
        urev = u[::-1]
        return sig.convolve(d, urev, mode="valid")


def marginals(node_potentials, count_potential, brute_force=False, VERBOSE=False,
              print_messages=False):
    Z = 0
    D = node_potentials.shape[0]
    marginals = np.zeros(D)
    ct_marginals = np.zeros(D+1)
    
    if brute_force:
        for idx in range(2**D):
            assn = idx_to_assn(idx, D)
            nen = neg_energy(assn, node_potentials, count_potential)
            
            for d in range(D):
                if assn[d] == 1:  marginals[d] += np.exp(nen)
            ct_marginals[np.sum(assn)] += np.exp(nen)
            Z += np.exp(nen)

        print "Z=", Z
        ct_marginals /= Z
        marginals /= Z

    else:
        marginals, ct_marginals = pass_all_messages(np.exp(node_potentials), np.exp(count_potential),
                                                    print_messages=print_messages)
        

    if VERBOSE:
        print "Z =", Z
        for idx in range(2**D):
            assn = idx_to_assn(idx, D)
            nen = neg_energy(assn, node_potentials, count_potential)
            print "%s\t%4f" % (assn, np.exp(nen)/Z)
        print "marginals", marginals

    return marginals, ct_marginals


def independent_sample(node_potentials, count_potential, brute_force=True):
    """ Draw samples from p(hj|v) for each j independently. """

    D = node_potentials.shape[0]

    qs, ct_margs = marginals(node_potentials, count_potential, brute_force=brute_force)

    return np.int32(np.random.rand(D) < qs)




def joint_sample(node_potentials, count_potential, brute_force=True):
    """ Draw a joint sample from p(h|v). """
    
    if brute_force:
        Z = 0
        D = node_potentials.shape[0]
        joint_probs = np.zeros(2**D)
        for idx in range(2**D):
            assn = idx_to_assn(idx, D)
            nen = neg_energy(assn, node_potentials, count_potential)
            joint_probs[idx] = np.exp(nen)
            Z += np.exp(nen)
        joint_probs /= Z

        idx_arr = np.nonzero(np.random.multinomial(1, joint_probs) == 1)[0]
        return idx_to_assn(idx_arr[0], D)

    else:

        return backward_messages_forward_sample(np.exp(node_potentials),
                                                np.exp(count_potential), n=1)
        

def compute_bmessage(exp_node_potential, h, result_matrix, result_row, count_cap=None):
    """ Compute backward message and put result in result_matrix[result_row,:].
    Do it this way so that we don't allocate new memory.

    Assumes z_d was defined as z_d = y_d + z_{d+1}. """
    
    result_matrix[result_row,:]   = h
    result_matrix[result_row,1:] += exp_node_potential * h[:-1]
    #result_matrix[result_row,count_cap+1:] = 0

    # normalize for numerical stability
    result_matrix[result_row,:] /= np.sum(result_matrix[result_row,:])


def compute_fmessage(exp_node_potential, h, result_matrix, result_row, count_cap=None):
    """ Compute forward message and put result in result_matrix[result_row,:].
    Do it this way so that we don't allocate new memory.

    Assumes z_d was defined as z_d = y_d + z_{d+1}. """

    result_matrix[result_row,:]             = exp_node_potential * np.hstack([h[1:], 0])

    #result_matrix[result_row,:-result_row] += h[:-result_row]
    result_matrix[result_row,:] += h

    # normalize for numerical stability
    result_matrix[result_row,:] /= np.sum(result_matrix[result_row,:])


def pass_all_messages(exp_node_potentials, exp_count_potential, count_cap=None,
                      print_messages=False):
    D = exp_node_potentials.shape[0]

    if count_cap is None:
        count_cap = np.max(np.nonzero(exp_count_potential > 1e-20))
        #count_cap = (np.cumsum(exp_count_potential)*(exp_count_potential>1e-20)).argmax()
        #print "Count cap =", count_cap

    #fmsgs = np.zeros((D+1,D+1))  # forward messages
    #bmsgs = np.zeros((D+1,D+1))  # backward messages

    fmsgs = np.zeros((D+1,count_cap+1))  # forward messages
    bmsgs = np.zeros((D+1,count_cap+1))  # backward messages

    fmsgs[0, :] = exp_count_potential[:count_cap+1]
    for d in range(D-1):
        compute_fmessage(exp_node_potentials[d],   fmsgs[d,:],   fmsgs, d+1, count_cap=count_cap)

    bmsgs[D,:2] = [1, exp_node_potentials[D-1]]
    for d in reversed(range(1,D)):
        compute_bmessage(exp_node_potentials[d-1], bmsgs[d+1,:], bmsgs, d, count_cap=count_cap)

    if print_messages:
        print "forward messages"
        print fmsgs
        print "backward messages"
        print bmsgs

    count_beliefs = exp_count_potential[:count_cap+1] * bmsgs[1,:]
    count_marginals = np.zeros(D+1)
    count_marginals[:count_cap+1] = count_beliefs / np.sum(count_beliefs)

    # construct pairwise beliefs (without explicitly instantiating the D^2
    # size matrices), then sum the diagonal to get b0, and the off-diagonal
    # to get b1.  b1/(b0+b1) gives marginal for original y_d for all except
    # the last variable, y_D.  we need to special case it, because there is
    # no pairwise potential that represents \theta_D -- it's just a unary in
    # the transformed model.
    bb = bmsgs[2:,:]
    ff = fmsgs[:-2,:]
    b0 = np.sum(bb*ff,axis=1)
    b1 = np.sum(bb[:,:-1]*ff[:,1:], axis=1) * exp_node_potentials[:-1]

    marginals = np.zeros(D)
    marginals[:-1] = b1/(b0+b1)

    # could probably structure things so the Dth var doesn't need to be
    # special-cased.  but this will do for now.  rather than computing
    # a belief at a pairwise potential, we do it at the variable.
    b0_D = fmsgs[D-1,0]*bmsgs[D,0]
    b1_D = fmsgs[D-1,1]*bmsgs[D,1]
    marginals[D-1] = b1_D / (b0_D+b1_D)

    VERBOSE = False
    if VERBOSE:
        print "BP marginals"
        print marginals
        print
        print "forward/backward msgs"
        print fmsgs
        print bmsgs
        print
        print "BP count marginals", count_beliefs / np.sum(count_beliefs)
        print "BPZ", np.sum(count_beliefs)
        print

    return marginals, count_marginals


def backward_messages_forward_sample(exp_node_potentials, exp_count_potential, n=50000, count_cap=None):
    D = exp_node_potentials.shape[0]
    if count_cap is None:
        count_cap = np.max(np.nonzero(exp_count_potential > 1e-20))
        #print "Count cap =", count_cap

    bmsgs = np.zeros((D+1,count_cap+1))

    bmsgs[D,:2] = [1, exp_node_potentials[D-1]]
    for d in reversed(range(1,D)):
        compute_bmessage(exp_node_potentials[d-1], bmsgs[d+1,:], bmsgs, d, count_cap=count_cap)

    count_beliefs = exp_count_potential[:count_cap+1] * bmsgs[1,:]

    # could repeat the following many times if desired
    count_freq = np.zeros(D+1)
    marginals = np.zeros(D)
    NUM_SAMPLES = n
    for i in range(NUM_SAMPLES):
        sample = np.zeros(D)
        D_on = random_categorical(count_beliefs/np.sum(count_beliefs))
        z_d = D_on
        for d in range(D-1):
            if z_d == 0:  break

            # unnormalized conditional probs p(z_d-1|z_d) = p(y_d=1|z_d) using backwards
            # messages to compute the reparameterization.
            p1 = exp_node_potentials[d] * bmsgs[d+2,z_d-1]  
            p0 = bmsgs[d+2,z_d]
            
            sample[d] = np.random.rand() < p1 / (p0 + p1)
            z_d -= sample[d]

        sample[D-1] = z_d
        
        assert D_on == np.sum(sample)

        if n == 1:  return sample

        count_freq[np.sum(sample)] += 1
        marginals += sample

    print "Drew %s samples..." % n
    print "Back-msgs, forward-sample count marginals"
    print count_freq / np.sum(count_freq)

    print "Back-msgs, forward-sample  marginals"
    print marginals / NUM_SAMPLES


def assignment_to_hard_node_pots(assn):
    """ assn is assumed binary """

    D = assn.shape[0]
    exp_node_pots = np.zeros(D)
    for d in range(D):
        if assn[d] == 0:  exp_node_pots[d] = 0
        else:             exp_node_pots[d] = np.inf

    return exp_node_pots


if __name__ == "__main__":
    D = int(sys.argv[1])

    np.random.seed(0)
    
    node_potentials = np.log((.0001+np.arange(D))/D) #np.random.randn(D)

    count_potential = -np.inf * np.ones(D+1)
    count_potential[2:5] = 0
    #count_potential = np.ones(D+1)
    #count_potential = np.random.randn(D+1) #np.zeros(D+1)
    #count_potential[3:] = -np.inf
    #count_potential[2] = np.log(10000)

    if False:
        A = np.zeros(7)
        B = np.zeros(5)
        C = np.zeros(3)
        
        a = np.array([1,10])
        b = np.array([1,2,3])
        c = np.array([3,4])
        A[:a.shape[0]] = a
        B[:b.shape[0]] = b
        C[:c.shape[0]] = c
        res1 = np.convolve(A, B[::-1], mode="valid")
        res2 = np.convolve(A, C[::-1], mode="valid")
        
        print 80 * "*"
        print A, B
        print res1
        print A, C
        print res2
        print 80 * "*"
        print
        
        res1 = np.convolve(a, b[::-1], mode="full")
        res2 = np.convolve(a, c[::-1], mode="full")
        
        print 80 * "*"
        print a, b
        print res1
        print a, c
        print res2
        print 80 * "*"
        
        print
        sys.exit(0)

    print
    print 80 * "*"
    print
    print "NP", node_potentials
    print "CP", count_potential
    print
    print 80 * "*"
    print

    CAN_BRUTE_FORCE = D <= 16
    
    if CAN_BRUTE_FORCE:
        qs, ct_margs = marginals(node_potentials, count_potential)
        print "Exact marginals (brute force)"
        print qs
        print "Exact count marginals (brute force)"
        print ct_margs
        print
    
    qs, ct_margs = marginals(node_potentials, count_potential, brute_force=False, print_messages=True)
    print "Exact marginals (chain BP)"
    print qs
    print "Exact count marginals (chain BP)"
    print ct_margs

    print
    
    T = make_balanced_binary_tree(D)
    # view tree with:  scipy.cluster.hierarchy.dendrogram(T)

    # count potentials are stored in a dictionary, where internal node indexes map
    # to count potentials
    exp_count_pot_dict = {}  # dictionary mapping internal node indexes to exp count potentials
    root_idx = D + T.shape[0]-1
    exp_count_pot_dict[root_idx] = np.exp(count_potential)

    fft_margs, fft_ct_margs, fft_log_Z = conv_tree(np.exp(node_potentials), exp_count_pot_dict, T,
                                                   use_fft=False)
                             
    print "Exact marginals (FFT)"
    print fft_margs

    print "Exact count marginals (FFT)"
    print fft_ct_margs[root_idx]

    fft_ct_at_root = fft_ct_margs[root_idx]
    print "FFT max error:", np.max(np.sqrt((fft_ct_at_root - ct_margs[:fft_ct_at_root.shape[0]])**2))

    print "Test backward-messages, forward-sample"
    backward_messages_forward_sample(np.exp(node_potentials), np.exp(count_potential),
                                     n=1000)
    
    sys.exit(0)

    if False:
        print "Independent samples"
        for k in range(0):
            sample = independent_sample(node_potentials, count_potential)
            print np.sum(sample), sample
        print

    if CAN_BRUTE_FORCE:
        print "Exact joint samples (brute force)"
        for k in range(10):
            sample = joint_sample(node_potentials, count_potential)
            print np.sum(sample), sample
        print

    print "Exact joint samples (BP)"
    for k in range(10):
        sample = joint_sample(node_potentials, count_potential, brute_force=False)
        print np.sum(sample), sample
    print    

    print "Test backward-messages, forward-sample"
    backward_messages_forward_sample(np.exp(node_potentials), np.exp(count_potential),
                                     n=10)
