#import deepxde as dde

import numpy as np
#import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

####################################################################################################################
####################################################################################################################

####################################################################################################################
def addWeights(loss_weights,weights):
    for w in weights:
        loss_weights.append(w)

    return loss_weights
####################################################################################################################
def transformTauAbs(X,Q):
    Q_t = tf.concat(
        [Q[:,0:1],
         Q[:,1:2],
         Q[:,2:3],
         tf.math.abs(Q[:,3:4]),
         Q[:,4:5],
         tf.math.abs(Q[:,5:6]),
         ], axis=1
        )

    return Q_t

def transformTauReLU(X,Q):
    Q_t = tf.concat(
        [Q[:,0:1],
         Q[:,1:2],
         Q[:,2:3],
         tf.nn.relu(Q[:,3:4]),
         Q[:,4:5],
         tf.nn.relu(Q[:,5:6]),
         ], axis=1
        )

    return Q_t

def transformScaling(X,Q):
    Q_t = tf.concat(
        [Q[:,0:1],
         Q[:,1:2],
         Q[:,2:3],
         100*Q[:,3:4],
         100*Q[:,4:5],
         100*Q[:,5:6],
         ], axis=1
        )

    return Q_t