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
from ctypes import *
import time
import sys
import random


np_double_type = np.double
c_int_p = POINTER(c_int)
c_double_p = POINTER(c_double)

KNCALIB__ = np.ctypeslib.load_library("knca_alg.A.dylib", ".")


class KNCAAlg:

    def __init__(self, batch_size, N, K, Kp, class_counts, C):
        self.N = N
        self.B = batch_size
        self.K = K
        self.Kp = Kp
        self.class_counts = class_counts
        self.C = C

        self.Infer = KNCALIB__["infer"]

        # reuse storage, so we don't keep re-allocating memory
        self.fmsgs1_c = (self.N * (self.K+1) * c_double)()
        self.bmsgs1_c = (self.N * (self.K+1) * c_double)()

        self.fmsgs2_c = ((self.C+1) * (self.K+1) * c_double)()
        self.bmsgs2_c = ((self.C+1) * (self.K+1) * c_double)()

        self.class_counts_c = class_counts.ctypes.data_as(c_double_p)

        self.result_marginals_c = (self.N * self.B * c_double)()
        self.result_log_Zs_c = (self.B * c_double)()


    def infer(self, exp_node_pots, Ys):
        exp_node_pots = (exp_node_pots[:]).astype(np_double_type)
        exp_node_pots_c = exp_node_pots.ctypes.data_as(c_double_p)

        # was having trouble passing this to c as an int.  just using
        # doubles to save some pointless debugging.
        Ys = Ys.astype(np.double)
        Ys_c = Ys.ctypes.data_as(c_double_p)

        #print "Class Counts"
        #print np.ctypeslib.as_array(self.class_counts_c)

        self.Infer(c_int(self.N), c_int(self.B), c_int(self.K), c_int(self.Kp),
                   self.class_counts_c, c_int(self.C), exp_node_pots_c, Ys_c,
                   self.fmsgs1_c, self.bmsgs1_c, self.fmsgs2_c, self.bmsgs2_c,
                   self.result_marginals_c, self.result_log_Zs_c)

        marginals = np.ctypeslib.as_array(self.result_marginals_c).reshape((self.B,self.N))
        log_Zs = np.ctypeslib.as_array(self.result_log_Zs_c).reshape((self.B))

        return marginals, log_Zs

    
if __name__ == "__main__":
    B    = 2  # batch size
    C    = 3  # num classes
    K    = 3  # num neighbors  
    Kp   = 2  # target class count

    Ys = np.array([0,0,0,1,1,2,2,2])
    N = Ys.shape[0]
    C = np.max(Ys)+1

    class_counts = np.array([np.sum(Ys == c) for c in range(C)]) #, dtype=np.double)
    
    print "Class counts", class_counts
    class_counts = class_counts.astype(np.double)

    exp_dists = np.zeros((B,N))
    for b in range(B):
        for n in range(N):
            exp_dists[b,n] = b+n+1
    
    knca = KNCAAlg(B, N, K, Kp, class_counts, C)

    marginals, log_Zs = knca.infer(exp_dists[:B,:], Ys)

    print "exp node pots"
    print exp_dists
    print "ys"
    print Ys
    
    print "marginals"
    print marginals
    print "log Zs"
    print log_Zs
