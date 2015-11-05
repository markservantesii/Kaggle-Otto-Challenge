from __future__ import division


import numpy as np
import pandas as pd

"""This file averages the predictions generated from multiple models. """

def load_data(filename):
    df = pd.read_csv(filename)
    X,ids = df.drop('id',axis=1),df['id']
    return X,ids

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

def combine_data(DF,ids,weights=None):
    sum = 0
    if weights is None:
        for df in DF:
            sum+=df
        sum = sum/len(DF)
    else:
        for i in range(len(DF)):
            sum+=DF[i]*weights[i]
    sum.insert(0,'id',value=ids)
    print("Data was combined.")
    return sum

def make_submission(df,filename):
    df.to_csv(filename,index=False,encoding = 'utf-8')
    print("Data saved to file.")

#=======================
# Script begins here
#=======================

X=[]


x0,ids = load_data("XGB-XT-final-5-17.csv")
x1,ids = load_data('NN13-final-5-18.csv')
x2,ids = load_data("XGB-XT-final2-5-17.csv")


x = [x0,x1]
print("Data was Loaded.")

for i in x:
    X.append(i)


X = combine_data(X,ids)

#This was the final submission that placed me in the top 10%
make_submission(X,"sub3-5-18.csv")
