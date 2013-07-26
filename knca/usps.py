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

from scipy.io import loadmat
import numpy as np
import Image

raw = loadmat('../data/usps_all.mat')['data']
classes = [None] * 10

dim = raw.shape[0]
num_digits_per_class = raw.shape[1]
num_classes = raw.shape[2]

def get_usps_split_corrupt(seed,digits=range(num_classes),noise=0):
    X,y = get_usps_split(seed,digits)
    Xtrain = X[0]
    Xtest = X[1]
    ytrain = y[0]
    ytest = y[1]
    p = np.random.permutation(ytrain.shape[0])[0:int(noise*ytrain.shape[0])]
    ytrain[p] = np.random.randint(0,ytrain.max()+1,p.shape[0])
    D = {'Xtrain':Xtrain,'Xtest':Xtest,'ytrain':ytrain,'ytest':ytest}

    return D

def get_usps_split(seed,digits=range(num_classes)):
    from numpy.random import RandomState
    rnd = RandomState(seed)

    num_train = 200
    num_test = 500

    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    for t in digits:
        I = rnd.permutation(num_digits_per_class)

        ## note: num_train + num_test < len(I);
        ## The NCA paper used 200 train and 500 test.
        ## This gives 700 per class. 
        I_train = I[:num_train]
        I_test = I[-num_test:]

        X_t_train = raw[:,I_train,t].T
        X_t_test = raw[:,I_test,t].T

        X_train.extend(X_t_train)
        X_test.extend(X_t_test)
        Y_train.extend([t] * num_train)
        Y_test.extend([t] * num_test)

    assert len(X_train) == len(Y_train) and len(X_test) == len(Y_test)
    
    # note: we only permute the training cases, since we don't do 
    # sgd of any kind on the test cases.

    import numpy as np

    I = rnd.permutation(len(X_train))
    X_train = np.array(X_train)[I]
    Y_train = np.array(Y_train)[I]

    I = rnd.permutation(len(X_test))
    X_test = np.array(X_test)[I]
    Y_test = np.array(Y_test)[I]

    X_train = usps_resizer(X_train,8)
    X_test = usps_resizer(X_test,8)

    X_train /= 255.
    X_test /= 255.

    return (X_train,X_test),(Y_train,Y_test)

def usps_resizer(X, newdim):
    Xnew = np.zeros((X.shape[0],newdim**2))
    for i in range(X.shape[0]):
        x = X[i,:].reshape((16,16))
        img = Image.fromarray(x)
        img = img.transpose(4).transpose(0).resize((newdim,newdim), Image.ANTIALIAS)
        Xnew[i,:] = np.fromstring(img.tostring(),np.uint8)
    return Xnew