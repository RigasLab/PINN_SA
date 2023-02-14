import numpy as np
#############################################
#############################################
def func_zeros(X):
    return 0*X[:, 0:1]

def func_const(val):
    def output(X):
        return 0*X[:, 0:1] + val

    return output
############################################
def addWeights(loss_weights,weights):
    for w in weights:
        loss_weights.append(w)

    return loss_weights
#############################################
