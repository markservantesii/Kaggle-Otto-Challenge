from __future__ import absolute_import
from __future__ import print_function


"""This file contains code for creating Neural Nets using keras. The predictions are saved to file."""
import numpy as np
import pandas as pd
from random import randrange,random,uniform

from otto_funs import logloss_mc


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, Adadelta, Adagrad


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

'''
    This demonstrates how to reach a score of 0.4890 (local validation)
    on the Kaggle Otto challenge, with a deep net using Keras.
    Compatible Python 2.7-3.4
    Recommended to run on GPU:
        Command: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python kaggle_otto_nn.py
        On EC2 g2.2xlarge instance: 19s/epoch. 6-7 minutes total training time.
    Best validation score at epoch 21: 0.4881
    Try it at home:
        - with/without BatchNormalization (BatchNormalization helps!)
        - with ReLU or with PReLU (PReLU helps!)
        - with smaller layers, largers layers
        - with more layers, less layers
        - with different optimizers (SGD+momentum+decay is probably better than Adam!)
'''

#np.random.seed(1337) # for reproducibility

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X) # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids

def preprocess_data(X, scaler=None):
    X = np.sqrt(X + 3/8.)
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def preprocess_labels(y, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)

       # y = y.astype(np.int32)
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


print("Loading data...")
X, labels = load_data('train.csv', train=True)
X_val, labels_val = load_data('eval.csv', train =True)
X, scaler= preprocess_data(X)
X_val, scaler_val = preprocess_data(X_val)

y, encoder = preprocess_labels(labels)
y_val, encoder_val = preprocess_labels(labels_val,categorical=False)

X_test, ids = load_data('test.csv', train=False)
X_test, _ = preprocess_data(X_test, scaler)

nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')

#-----------------------------
# Build 5 models and average
#----------------------------
proba1,proba2,cal_prob1,cal_prob2=0,0,0,0
for i in range(2):
    seed = randrange(1,100000)
    np.random.seed(seed)
    print("Building model %s." % i)

    '''
    ## NN0
    model1 = Sequential()

    model1.add(Dense(dims, 1024, init='glorot_uniform'))
    model1.add(PReLU((1024,)))
    model1.add(BatchNormalization((1024,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(1024, 512, init='glorot_uniform'))
    model1.add(PReLU((512,)))
    model1.add(BatchNormalization((512,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(512, 256, init='glorot_uniform'))
    model1.add(PReLU((256,)))
    model1.add(BatchNormalization((256,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(256, nb_classes, init='glorot_uniform'))
    model1.add(Activation('softmax'))

    model1.compile(loss='categorical_crossentropy', optimizer="adam")


    #NN1
    model1 = Sequential()

    model1.add(Dense(dims, 256, init='glorot_uniform'))
    model1.add(PReLU((256,)))
    model1.add(BatchNormalization((256,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(256, 512, init='glorot_uniform'))
    model1.add(PReLU((512,)))
    model1.add(BatchNormalization((512,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(512, 256, init='glorot_uniform'))
    model1.add(PReLU((256,)))
    model1.add(BatchNormalization((256,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(256, nb_classes, init='glorot_uniform'))
    model1.add(Activation('softmax'))


    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model1.compile(loss='categorical_crossentropy', optimizer=sgd)


    #NN2
    model1 = Sequential()

    model1.add(Dense(dims, 1024, init='glorot_uniform'))
    model1.add(PReLU((1024,)))
    model1.add(BatchNormalization((1024,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(1024, 256, init='glorot_uniform'))
    model1.add(PReLU((256,)))
    model1.add(BatchNormalization((256,)))
    model1.add(Dropout(0.3))

    model1.add(Dense(256, nb_classes, init='glorot_uniform'))
    model1.add(Activation('softmax'))


    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model1.compile(loss='categorical_crossentropy', optimizer=sgd)

    #NN4
    # This did not perform very well
    model1 = Sequential()

    model1.add(Dense(dims, 1024, init='glorot_uniform'))
    model1.add(PReLU((1024,)))
    model1.add(BatchNormalization((1024,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(1024, 128, init='glorot_uniform'))
    model1.add(PReLU((128,)))
    model1.add(BatchNormalization((128,)))
    model1.add(Dropout(0.3))

    model1.add(Dense(128, 128, init='glorot_uniform'))
    model1.add(PReLU((128,)))
    model1.add(BatchNormalization((128,)))
    model1.add(Dropout(0.3))

    model1.add(Dense(128, 64, init='glorot_uniform'))
    model1.add(PReLU((64,)))
    model1.add(BatchNormalization((64,)))
    model1.add(Dropout(0.3))

    model1.add(Dense(64, nb_classes, init='glorot_uniform'))
    model1.add(Activation('softmax'))


    sgd = SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True)
    model1.compile(loss='categorical_crossentropy', optimizer=sgd)
    '''
    '''
    ## NN3
    model1 = Sequential()

    model1.add(Dense(dims, 1024, init='glorot_uniform'))
    model1.add(PReLU((1024,)))
    model1.add(BatchNormalization((1024,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(1024, 512, init='glorot_uniform'))
    model1.add(PReLU((512,)))
    model1.add(BatchNormalization((512,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(512, nb_classes, init='glorot_uniform'))
    model1.add(Activation('softmax'))

    #sgd = SGD(lr=0.2, decay=1e-6, momentum=0.9, nesterov=True)
    model1.compile(loss='categorical_crossentropy', optimizer="adam")

    print("Training model %s." % i)

    model1.fit(X, y, nb_epoch=20, batch_size=32, validation_split=0.15)
    proba1 += model1.predict_proba(X_test)
    '''

    '''
    ## NN4 -- This did well ~0.48 on CV
    model1 = Sequential()


    model1.add(Dense(dims, 1024, init='glorot_uniform'))
    model1.add(Dropout(0.05))
    model1.add(PReLU((1024,)))
    model1.add(BatchNormalization((1024,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(1024, 512, init='glorot_uniform'))
    model1.add(PReLU((512,)))
    model1.add(BatchNormalization((512,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(512, nb_classes, init='glorot_uniform'))
    model1.add(Activation('softmax'))

    #sgd = SGD(lr=0.2, decay=1e-6, momentum=0.9, nesterov=True)
    model1.compile(loss='categorical_crossentropy', optimizer="adam")

    print("Training model %s." % i)

    model1.fit(X, y, nb_epoch=20, batch_size=128, validation_split=0.15)
    proba1 += model1.predict_proba(X_test)
    '''
    '''
    ## NN5 -- performed well on LB when averaged with NN4
    model1 = Sequential()


    model1.add(Dense(dims, 1024, init='glorot_uniform'))
    model1.add(Dropout(0.05))
    model1.add(PReLU((1024,)))
    model1.add(BatchNormalization((1024,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(1024, 512, init='glorot_uniform'))
    model1.add(PReLU((512,)))
    model1.add(BatchNormalization((512,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(512, 256, init='glorot_uniform'))
    model1.add(PReLU((256,)))
    model1.add(BatchNormalization((256,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(256, nb_classes, init='glorot_uniform'))
    model1.add(Activation('softmax'))

    #sgd = SGD(lr=0.2, decay=1e-6, momentum=0.9, nesterov=True)
    ada = Adagrad(lr=0.01, epsilon=1e-6)
    model1.compile(loss='categorical_crossentropy', optimizer="ada")

    print("Training model %s." % i)

    model1.fit(X, y, nb_epoch=20, batch_size=128, validation_split=0.15)
    proba1 += model1.predict_proba(X_test)
    '''
    '''
    ## NN6 -- Same as NN5  but with Adagrad
    model1 = Sequential()


    model1.add(Dense(dims, 1024, init='glorot_uniform'))
    model1.add(Dropout(0.05))
    model1.add(PReLU((1024,)))
    model1.add(BatchNormalization((1024,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(1024, 512, init='glorot_uniform'))
    model1.add(PReLU((512,)))
    model1.add(BatchNormalization((512,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(512, 256, init='glorot_uniform'))
    model1.add(PReLU((256,)))
    model1.add(BatchNormalization((256,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(256, nb_classes, init='glorot_uniform'))
    model1.add(Activation('softmax'))

    ada = Adagrad(lr=0.01, epsilon=1e-6)
    model1.compile(loss='categorical_crossentropy', optimizer=ada)

    print("Training model %s." % i)

    model1.fit(X, y, nb_epoch=20, batch_size=128, validation_split=0.15)
    proba1 += model1.predict_proba(X_test)
    '''
    '''
    ## NN7/NN8 -- Same as NN6  but with dropout on input
    ## NN7 uses adadelta, NN8 uses adam and has 21 epochs
    ## NN9 uses sqrt(X) transformation during preprocessing
    model1 = Sequential()

    model1.add(Dropout(0.05))
    model1.add(Dense(dims, 1024, init='glorot_uniform'))
    model1.add(PReLU((1024,)))
    model1.add(BatchNormalization((1024,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(1024, 512, init='glorot_uniform'))
    model1.add(PReLU((512,)))
    model1.add(BatchNormalization((512,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(512, 256, init='glorot_uniform'))
    model1.add(PReLU((256,)))
    model1.add(BatchNormalization((256,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(256, nb_classes, init='glorot_uniform'))
    model1.add(Activation('softmax'))

    #ada = Adagrad(lr=0.01, epsilon=1e-6)
    model1.compile(loss='categorical_crossentropy', optimizer="adam")

    print("Training model %s." % i)

    model1.fit(X, y, nb_epoch=21, batch_size=128, validation_split=0.20)
    proba1 += model1.predict_proba(X_test)
    '''
    ## NN10 uses sqrt(X) transformation during preprocessing
    ##      4 layers, 1024 nodes each, 2nd layer is sparse

    ## Creates a sparse matrix of weights (mostly zeros)
    ## to create a sparse connected layer (most connection weights are zero)
    W= np.zeros((93,1024))
    r = int(uniform(0.15,0.35)*93*1024)
    eps1 = randrange(60,75)
    eps2 = randrange(35,50)
    for i in range(r):
        W[randrange(93),randrange(1024)] = 1
    print(W.shape)
    '''
    ## model1
    model1 = Sequential()

    model1.add(Dropout(0.05))
    model1.add(Dense(dims, 256, init='glorot_uniform'))
    model1.add(PReLU((256,)))
    model1.add(BatchNormalization((256,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(256,128, init='glorot_uniform'))
    model1.add(PReLU((128,)))
    model1.add(BatchNormalization((128,)))
    model1.add(Dropout(0.25))


    model1.add(Dense(128, nb_classes, init='glorot_uniform'))
    model1.add(Activation('softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer="adam")

    model1.fit(X, y, nb_epoch=eps1, batch_size=64, validation_split=0.15)

    ## model 2
    model2 = Sequential()

    #model2.add(Dropout(0.05))
    model2.add(Dense(dims, 1024, init='glorot_uniform',weights=[W]))
    model2.add(PReLU((1024,)))
    model2.add(BatchNormalization((1024,)))
    model2.add(Dropout(0.5))

    model2.add(Dense(1024, 1024, init='glorot_uniform'))
    model2.add(PReLU((1024,)))
    model2.add(BatchNormalization((1024,)))
    model2.add(Dropout(0.5))

    model2.add(Dense(1024, 1024, init='glorot_uniform'))
    model2.add(PReLU((1024,)))
    model2.add(BatchNormalization((1024,)))
    model2.add(Dropout(0.5))

    model2.add(Dense(1024, nb_classes, init='glorot_uniform'))
    model2.add(Activation('softmax'))

    #ada = Adagrad(lr=0.01, epsilon=1e-6)
    model2.compile(loss='categorical_crossentropy', optimizer="adam")

    #print("Training model %s." % i)

    model2.fit(X,y,nb_epoch=eps2,batch_size=128,validation_split=.15)

    proba1 += model1.predict_proba(X_test)
    proba2 += model2.predict_proba(X_test)

    cal_prob1 += model1.predict_proba(X_val)
    cal_prob2 += model2.predict_proba(X_val)
    '''
    model1 = Sequential()

    model1.add(Dropout(0.15))
    model1.add(Dense(dims, 512, init='glorot_uniform'))
    model1.add(PReLU((512,)))
    model1.add(BatchNormalization((512,)))
    model1.add(Dropout(0.2))

    model1.add(Dense(512,512, init='glorot_uniform'))
    model1.add(PReLU((512,)))
    model1.add(BatchNormalization((512,)))
    model1.add(Dropout(0.2))


    model1.add(Dense(512, nb_classes, init='glorot_uniform'))
    model1.add(Activation('softmax'))

    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.7, nesterov=True)

    model1.compile(loss='categorical_crossentropy', optimizer="adam")

    model1.fit(X, y, nb_epoch=eps1, batch_size=128, validation_split=0.001)
#------------------------------------------------
## model1
    '''
    model1 = Sequential()

    model1.add(Dropout(0.10))
    model1.add(Dense(dims, 512, init='glorot_uniform'))
    model1.add(PReLU((512,)))
    model1.add(BatchNormalization((512,)))
    model1.add(Dropout(0.5))

    model1.add(Dense(512,512, init='glorot_uniform'))
    model1.add(PReLU((512,)))
    model1.add(BatchNormalization((512,)))
    model1.add(Dropout(0.5))


    model1.add(Dense(512, nb_classes, init='glorot_uniform'))
    model1.add(Activation('softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer="adam")

    model1.fit(X, y, nb_epoch=eps1, batch_size=64, validation_split=0.15)
    '''
    ## model 2
    model2 = Sequential()

    #model2.add(Dropout(0.05))
    model2.add(Dense(dims, 1024, init='glorot_uniform',weights=[W]))
    model2.add(PReLU((1024,)))
    model2.add(BatchNormalization((1024,)))
    model2.add(Dropout(0.5))

    model2.add(Dense(1024, 512, init='glorot_uniform'))
    model2.add(PReLU((512,)))
    model2.add(BatchNormalization((512,)))
    model2.add(Dropout(0.5))

    model2.add(Dense(512, 256, init='glorot_uniform'))
    model2.add(PReLU((256,)))
    model2.add(BatchNormalization((256,)))
    model2.add(Dropout(0.5))

    model2.add(Dense(256, nb_classes, init='glorot_uniform'))
    model2.add(Activation('softmax'))

    #ada = Adagrad(lr=0.01, epsilon=1e-6)
    model2.compile(loss='categorical_crossentropy', optimizer="adam")

    #print("Training model %s." % i)

    model2.fit(X,y,nb_epoch=eps2,batch_size=128,validation_split=.001)

    proba1 += model1.predict_proba(X_test)
    proba2 += model2.predict_proba(X_test)


    #cal_prob1 += model1.predict_proba(X_val)
    #cal_prob2 += model2.predict_proba(X_val)




    ## Can't get logloss to work on keras submissions
    '''
    cal_probs = (cal_prob1 + cal_prob2)/2.
    print("Model 1 loss: %0.4f" % logloss_mc(y_val,cal_prob1))
    print("Model 2 loss: %0.4f" % logloss_mc(y_val,cal_prob2))
    print("Avg loss: %0.4f" % logloss_mc(y_val,cal_probs))
    #proba_val += model1.predict_proba(X_val)
    '''


print("Generating submission...")
#proba = model.predict_proba(X_test)
proba1 = proba1/2.
proba2 = proba2/2.
#cal_prob1 = cal_prob1/2.
#cal_prob2 = cal_prob2/2.

avg_test = (proba1 + proba2)/2.
#avg_cal = (cal_prob1 + cal_prob2)/2.
'''
make_submission(proba1, ids, encoder, fname='NN1-test-5-16.csv')
make_submission(proba2, ids, encoder, fname='NN2-test-5-16.csv')
make_submission(cal_prob1, ids, encoder, fname='NN1-eval-5-16.csv')
make_submission(cal_prob2, ids, encoder, fname='NN2-eval-5-16.csv')

make_submission(avg_test, ids, encoder, fname='NN12-test-5-16.csv')
'''
make_submission(avg_test, ids, encoder, fname='NN13-final-5-18.csv')

