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

CHAINLIB__ = np.ctypeslib.load_library("chain/chain_alg.A.dylib", ".")


class ChainAlg:

    def __init__(self, num_hiddens, Kmin, Kmax, minibatch_size):
        self.D = num_hiddens
        self.B = minibatch_size
        self.Kmin = Kmin
        self.Kmax = Kmax
        self.K = Kmax+1
        
        self.Infer = CHAINLIB__["infer"]

        # reuse storage, so we don't keep re-allocating memory
        self.fmsgs_c = (self.D * self.K * c_double)()
        self.bmsgs_c = (self.D * self.K * c_double)()
        self.buf_c = ((self.D+1) * c_double)()

        self.result_marginals0_c = (self.D * self.B * c_double)()
        self.result_samples0_c = (self.D * self.B * c_byte)()

        self.result_logZs_c = (self.B * c_double)()
        self.result_marginals1_c = (self.D * self.B * c_double)()
        self.result_samples1_c = (self.D * self.B * c_byte)()


    def infer(self, minibatch_exp_node_pots, storage=0):
        exp_node_pots = (minibatch_exp_node_pots[:]).astype(np_double_type)
        exp_node_pots_c = exp_node_pots.ctypes.data_as(c_double_p)

        if storage == 0:
            self.Infer(c_int(self.D), c_int(self.Kmin), c_int(self.Kmax), c_int(self.B),
                       exp_node_pots_c, self.fmsgs_c, self.bmsgs_c, self.buf_c,
                       self.result_marginals0_c, 
                       self.result_logZs_c,
                       self.result_samples0_c,)

            marginals = np.ctypeslib.as_array(self.result_marginals0_c).reshape((self.B,self.D))
            logZs = np.ctypeslib.as_array(self.result_logZs_c).reshape((self.B))
            
            samples = np.ctypeslib.as_array(self.result_samples0_c).reshape((self.B,self.D))
        else:
            self.Infer(c_int(self.D), c_int(self.Kmin), c_int(self.Kmax), c_int(self.B),
                       exp_node_pots_c, self.fmsgs_c, self.bmsgs_c, self.buf_c,
                       self.result_marginals1_c,
                    self.result_logZs_c,
                self.result_samples1_c)

            logZs = np.ctypeslib.as_array(self.result_logZs_c).reshape((self.B))

            marginals = np.ctypeslib.as_array(self.result_marginals1_c).reshape((self.B,self.D))
            samples = np.ctypeslib.as_array(self.result_samples1_c).reshape((self.B,self.D))
            
        return marginals, samples, logZs

    
if __name__ == "__main__":
    D    = int(sys.argv[1]) 
    B    = int(sys.argv[2])

    Kmin = 1
    Kmax = 5

    node_potentials = np.zeros((B,D))
    for b in range(B):
        node_potentials[b,:] = np.log((.0001+np.arange(D))/D)
    
    ca = ChainAlg(D, Kmin, Kmax, B)

    marginals, samples = ca.infer(np.exp(node_potentials))

    # When D=8, marginals should be:
    # [[  1.46962972e-05   1.33622689e-01   2.43589030e-01   3.34093911e-01
    #     4.08760176e-01   4.70653314e-01   5.22281463e-01   5.65595412e-01]

    print marginals
    print np.mean(samples,axis=0)
