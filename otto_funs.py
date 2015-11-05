from __future__ import absolute_import
from __future__ import print_function

"""
This file is a collection of functions used for the Otto Challenge competition on Kaggle.
"""

import numpy as np
import pandas as pd
import random
from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import LabelEncoder,StandardScaler
from scipy.optimize import minimize

from keras.utils import np_utils, generic_utils

##-----------------------
##  multiclass log loss
##-----------------------

def logloss_mc(y_true, y_prob, epsilon=1e-15):
    """ Multiclass logloss
    This function is not officially provided by Kaggle, so there is no
    guarantee for its correctness.
    """
    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll


def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X)
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        print("--Loaded Data")
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        print("--Loaded Data")
        return X, ids

def preprocess_data(X, scaler=None,transform=True):
    if transform is True:
        X = np.sqrt(X+3/8.)
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)


    return X, scaler

def preprocess_labels(labels, encoder=None, NN=False):
    """convert y strings to numeric
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    y = label_encoder.fit_transform(y).astype(np.int32)"""

    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.fit_transform(labels).astype(np.int32)

# This expands the 'y' array to a matrix form; necessary to be used with keras
    if NN:
        y = np_utils.to_categorical(y)
    return y, encoder

def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))

def train_cal_val(X,y,train_size=1,NN=False):
    random.seed(111)
    X_train,X1,y_train,y1 = train_test_split(X,y,train_size = train_size)
    if NN:
        y_train_nn = np_utils.to_categorical(y_train)
        y_cal_nn =np_utils.to_categorical(y_cal)
        y_val_nn =np_utils.to_categorical(y_val)
        return X_train,y_train,y_train_nn,X_cal,y_cal,y_cal_nn,X_val,y_val,y_val_nn
    else:
        return X_train,y_train,X1,y1

def get_append_probs(models,X,n_classes=9):
    probs = np.zeros((len(X),n_classes*len(models)))
    for m in models:
        probs= np.c_[probs,m.predict_proba(X)]
    return probs

def fit_models(models,X_train,y_train):
    print("--Fitting Models")
    fitted_models = []
    for m in models:
        m.fit(X_train,y_train)
        fitted_models.append(m)
    print("--Finished fitting models")
    return fitted_models

def get_probs(models,X):
    probs=[]
    for m in models:
        probs.append(m.predict_proba(X))
    return probs

##-----------------------
##  Combines probabilities of multiple models
##  via weighted average
##-----------------------

def blend_probs(probs,y_true):
    x0 =np.ones(len(probs))/(len(probs))
    bnds = tuple((0,1) for x in x0)
    cons = ({'type':'eq','fun':lambda x: 1-sum(x)})
    weights = minimize(fun,x0,(probs,y_true),method='SLSQP',bounds=bnds,constraints=cons).x
    for i in range(len(weights)):
        print("Model %s Weight: %0.4f" % (i,weights[i]))
    y_prob = 0
    for i in range(len(probs)):
        y_prob += probs[i]*weights[i]
    return y_prob,weights

##-----------------------
##  Predicts class probabilities for each model
##  Calculates logloss for each
##-----------------------
def get_loss(models,X_valid,y_valid):
    probs,loss= [],[]
    for m in models:
        probs.append(m.predict_proba(X_valid))

    for p in probs:
        loss.append(logloss_mc(y_valid,p))
    return loss


##-----------------------
##  Function to be optimized for linear combo of prediction probs
##-----------------------
def fun(c,probs,y_true):
    sum = 0
    for i in range(len(probs)):
        sum+= probs[i]*c[i]
    return logloss_mc(y_true,sum)

def combine_probs(probs,weights):
    sum=0
    for i in range(len(probs)):
        sum+= probs[i]*weights[i]
    return sum

