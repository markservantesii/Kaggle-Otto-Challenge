from __future__ import division
import otto_funs as of
import sys

"""This file is the primary code for building Tree-based classifiers. The predictions are averaged together
and saved in a single file."""




sys.path.append('c:/anaconda/xgboost/wrapper')
from XGBoostClassifier2 import XGBoostClassifier

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from numpy import random
from random import randrange
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU


#----------------------------------------
#   Build Many xgboost clf's
#----------------------------------------
def build_0(X_train,y_train):
    print("--Building and Training XGBoost Models")
    XB,XB_cal =[],[]


    seed=randrange(1,10000)
    #random.seed(seed)


    XB.append(XGBoostClassifier(n_iter=150,
                                max_features=0.3,
                                max_depth=7,
                                min_child_weight=10,
                                gamma=0.0093,random_state=seed,
                                learning_rate=0.2,
                                l2_weight=0.1,
                                max_samples=0.77,
                                )
              )
    XB.append(ExtraTreesClassifier(n_estimators=100,criterion="entropy",min_samples_split=1,random_state=seed))



    for xb in XB:
        xb_cal = CalibratedClassifierCV(base_estimator=xb,method='isotonic',cv=5).fit(X_train,y_train)
        XB_cal.append(xb_cal)
    return XB_cal

def build_1(X_train,y_train):
    print("--Building and Training XGBoost Models")
    XB,XB_cal =[],[]


    seed=randrange(1,10000)
    #random.seed(seed)

    XB.append(GradientBoostingClassifier(learning_rate=0.01,n_estimators=100,random_state=seed))
    XB.append(ExtraTreesClassifier(n_estimators=100,criterion="entropy",min_samples_split=1,random_state=seed))

    for xb in XB:
        xb_cal = CalibratedClassifierCV(base_estimator=xb,method='isotonic',cv=5).fit(X_train,y_train)
        XB_cal.append(xb_cal)
    return XB_cal

def build_XB(X_train,y_train,X_cal,y_cal,X_test):
    cal_prob,test_prob = 0,0
    for i in range(3):
        print("--Building and Training model %s" % i)
        seed = randrange(1,10000)
        model = XGBoostClassifier(n_iter=1000,
                                max_features=0.3,
                                max_depth=8,
                                min_child_weight=10,
                                gamma=0.0093,random_state=seed,
                                learning_rate=0.05,
                                l2_weight=0.1,
                                max_samples=0.77,
                                )
        model = CalibratedClassifierCV(base_estimator=model,method='isotonic',cv=3).fit(X_train,y_train)
        print("Model %s training complete." % i)
        test_prob += model.predict_proba(X_test)
    test_prob = test_prob/3.
    return(cal_prob,test_prob)

def build_XB1(X_train,y_train,X_cal,y_cal,X_test):
    cal_prob,test_prob = 0,0
    for i in range(5):
        print("--Building and Training model %s" % i)
        seed = randrange(1,10000)
        model = XGBoostClassifier(n_iter=500,
                                max_features=0.3,
                                max_depth=7,
                                min_child_weight=10,
                                gamma=0.0093,random_state=seed,
                                learning_rate=0.2,
                                l2_weight=0.1,
                                max_samples=0.9
                                )
        model = CalibratedClassifierCV(base_estimator=model,method='isotonic',cv=5).fit(X_train,y_train)
        print("Model %s training complete." % i)
        cal_prob += model.predict_proba(X_cal)
        test_prob += model.predict_proba(X_test)
    cal_prob = cal_prob/5.
    test_prob = test_prob/5.
    print("Average Model Loss: %0.4f" % of.logloss_mc(y_cal,cal_prob))
    return(cal_prob,test_prob)

def build_XT(X_train,y_train,X_cal,y_cal,X_test):
    cal_prob,test_prob = 0,0
    for i in range(3):
        print("--Building and Training model %s" % i)
        seed = randrange(1,10000)
        model = ExtraTreesClassifier(n_estimators=500,criterion="entropy",min_samples_split=1,n_jobs=-1,
                                     random_state=seed,max_features=0.9)
        model = CalibratedClassifierCV(base_estimator=model,method='isotonic',cv=5).fit(X_train,y_train)
        print("Model %s training complete." % i)
        test_prob += model.predict_proba(X_test)
    test_prob = test_prob/3.
    return(cal_prob,test_prob)

def build_XT1(X_train,y_train,X_cal,y_cal,X_test):
    cal_prob,test_prob = 0,0
    for i in range(3):
        print("--Building and Training model %s" % i)
        seed = randrange(1,10000)
        model = ExtraTreesClassifier(n_estimators=500,criterion="entropy",min_samples_split=1,random_state=seed,n_jobs=-1)
        model = CalibratedClassifierCV(base_estimator=model,method='isotonic',cv=5).fit(X_train,y_train)
        print("Model %s training complete." % i)
        test_prob += model.predict_proba(X_test)
    test_prob = test_prob/3.
    return(cal_prob,test_prob)

def build_NN(X_train,y_train_nn,X_cal,y_cal,X_test):
    cal_prob,test_prob = 0,0
    nb_classes = y_train_nn.shape[1]
    dims = X_train.shape[1]

    for i in range(5):
        print("--Building and Training model %s" % i)
        seed = randrange(1,10000)
        np.random.seed(seed)

        model = Sequential()

        model.add(Dense(dims, 512, init='glorot_uniform'))
        model.add(PReLU((512,)))
        model.add(BatchNormalization((512,)))
        model.add(Dropout(0.5))

        model.add(Dense(512, 512, init='glorot_uniform'))
        model.add(PReLU((512,)))
        model.add(BatchNormalization((512,)))
        model.add(Dropout(0.5))

        model.add(Dense(512, 512, init='glorot_uniform'))
        model.add(PReLU((512,)))
        model.add(BatchNormalization((512,)))
        model.add(Dropout(0.5))

        model.add(Dense(512, nb_classes, init='glorot_uniform'))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer="adam")

        model.fit(X_train, y_train_nn, nb_epoch=20, batch_size=16, validation_split=0.15)

        cal_prob += model.predict_proba(X_cal)
        test_prob += model.predict_proba(X_test)
    cal_prob,test_prob = cal_prob/10.,test_prob/10.
    print("Avg NN Model Loss: %0.4f" % of.logloss_mc(y_cal,cal_prob))
    return(cal_prob,test_prob)

def split_xb(XB):
    XB0,XB1 =[],[]
    for i,j in XB:
        XB0.append(i)
        XB1.append(j)
    return XB0,XB1

def runsum_probs(models,X_valid):
    sum=0
    for m in models:
        sum+=m.predict_proba(X_valid)
    return sum,len(models)

#----------------------------------------
#   Code for building many forests
#----------------------------------------
def build_rf(X_train,y_train):
    print("--Building and Training Model")
    seed = randrange(1,100000)


    rf = RandomForestClassifier(n_estimators=800,
                                n_jobs=-1,
                                criterion="entropy",
                                bootstrap=False,
                                random_state=seed,
                                #min_samples_split=randrange(1,3),
                                max_features=0.3
                                )

    rf_cal = CalibratedClassifierCV(base_estimator=rf,method='isotonic',cv=5).fit(X_train,y_train)
    print("--Model training complete.")
    return rf_cal

def keep_best_loss(X_train,y_train,X_valid,y_valid):
    PROB,LOSS,RF = [],[],[]
    for i in range(0,6):
        seed=randrange(1,10000)
        random.seed(seed)
        print("Seed: %s" % seed)
        rf = build_rf(X_train,y_train)
        prob = rf.predict_proba(X_valid)
        loss = of.logloss_mc(y_valid,prob)
        RF.append(rf)
        PROB.append(prob)
        LOSS.append(loss)
        print("Model %s Loss: %0.4f" % (i,loss))
    sum=0
    for p in PROB:
        sum+=p
    avg_prob = sum/len(PROB)
    avg_loss = of.logloss_mc(y_valid,avg_prob)
    print("Avg Loss of 5 models: %0.4f" % avg_loss)
    return RF,PROB

def build_many_models(n_models=30):
    RF,XB=[],[]
    for i in range(n_models):
        RF.append(RandomForestClassifier(n_estimators=100,
                                         n_jobs=-1,
                                         criterion="entropy",
                                         bootstrap="False",
                                         max_features=0.3
                                         )
                  )
        XB.append(XGBoostClassifier(n_iter=150,
                            max_features=0.3,
                            max_depth=7,
                            min_child_weight=10,
                            gamma=0.0093,random_state=1010,
                            learning_rate=0.2,
                            l2_weight=0.1,
                            max_samples=0.77,
                            )
                  )
    print("--Built Models")
    return RF,XB

def calibrate_models(RF,XB):
    rf_cal_models,xb_cal_models=[],[]
    print("--Calibrating Models")
    for rf in RF:
        rf_cal_models.append(CalibratedClassifierCV(base_estimator=rf,method='isotonic',cv=5))
    for xb in XB:
        xb_cal_models.append(CalibratedClassifierCV(base_estimator=xb,method='isotonic',cv=5))
    print("--Calibration complete.")
    return rf_cal_models,xb_cal_models

def fit_many_models(rf_models,xb_models,X_train,y_train):
    rf_fitted_models,xb_fitted_models=[],[]
    print("--Fitting Models")
    for rf in rf_models:
        rf_fitted_models.append(rf.fit(X_train,y_train))
        print("RF Model %s complete." % rf_models.index(rf))
    for xb in xb_models:
        xb_fitted_models.append(xb.fit(X_train,y_train))
        print("XB Model %s complete." % xb_models.index(xb))
    print("--Fitting Complete.")
    return rf_fitted_models,xb_fitted_models

def avg_probs(rf_models,X_val):
    probs=0
    print("--Calculating Average")
    for rf in rf_models:
        probs+= rf.predict_proba(X_val)
    avg = probs/len(rf_models)
    print("--Averaging Complete.")
    return avg

def build_models():
    #LR = LogisticRegression()
    xgb = XGBoostClassifier(n_iter=150,
                            max_features=0.3,
                            max_depth=7,
                            min_child_weight=10,
                            gamma=0.0093,random_state=1010,
                            learning_rate=0.2,
                            l2_weight=0.1,
                            max_samples=0.77,
                            )
    gb = GradientBoostingClassifier(learning_rate=0.01,n_estimators=50)
    xt = ExtraTreesClassifier(n_estimators=100,criterion="entropy",min_samples_split=1)
    rf = RandomForestClassifier(n_estimators=200,
                                bootstrap=False,
                                n_jobs=-1,
                                max_features=0.3,
                                criterion="entropy")

    rf1 = RandomForestClassifier(n_estimators=200,
                                bootstrap=False,
                                n_jobs=-1,
                                max_features=0.3,
                                min_samples_split=1,
                                criterion="entropy",
                                )
    rf2 = RandomForestClassifier(n_estimators=100,
                                bootstrap=True,
                                max_features=0.2,
                                n_jobs=-1,
                                criterion="entropy",
                                min_samples_split=1
                                )

    xg_cal = CalibratedClassifierCV(base_estimator = xgb,method='isotonic',cv=5)


    rf_cal = CalibratedClassifierCV(base_estimator = rf,method='isotonic',cv=5)
    gb_cal = CalibratedClassifierCV(base_estimator = gb,method='isotonic',cv=5)
    xt_cal = CalibratedClassifierCV(base_estimator = xt,method='isotonic',cv=5)

    #models=[xgb]
    models = [rf_cal,xt_cal,gb_cal,xg_cal]
    return models


#===================================
# Script Starts Here
#====================================

# Load and process data: X,y ; X_val,y_val ; X_test
## Training data, X,y
X,labels = of.load_data(path = "c:/users/mservant/kaggle/competitions/otto/train.csv",train=True)
X,scaler = of.preprocess_data(X)
y,encoder = of.preprocess_labels(labels)

## Test and eval data
#X_train,y_train,X_val,y_val = of.train_cal_val(X,y,NN=False)
X_test, ids = of.load_data(path = "c:/users/mservant/kaggle/competitions/otto/test.csv",train=False)
X_val, val_labels = of.load_data(path = "c:/users/mservant/kaggle/competitions/otto/eval.csv",train=True)
X_val,scaler_eval = of.preprocess_data(X_val)
y_val,val_encoder = of.preprocess_labels(val_labels)

X_test,_ = of.preprocess_data(X_test,scaler=scaler)

#-------------------------------
# Code for building 10 models of each type: XGBoost, ExtraTrees, NN
#--------------------------------

XB_cal_prob,XB_test_prob = build_XB(X,y,X_val,y_val,X_test)
XT_cal_prob,XT_test_prob = build_XT(X,y,X_val,y_val,X_test)
XT1_cal_prob,XT1_test_prob = build_XT1(X,y,X_val,y_val,X_test)
test_probs = [XB_test_prob,XT_test_prob,XT1_test_prob]

avg_test_probs = 0
for m in test_probs:
    avg_test_probs += m
avg_test_probs = avg_test_probs/len(test_probs)

print("Making submission file.")
of.make_submission(avg_test_probs,ids,encoder,"XGB-XT-final2-5-17.csv")


