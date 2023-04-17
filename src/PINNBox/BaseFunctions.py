import numpy as np
#############################################
#############################################
def func_zeros(X):
    """
    Return an array of zeros.

    :param X: Coordinates at which zeros are found.
    :type X: numpy array
    :return: Zeros matching number of points in X.
    :rtype: numpy array

    """
    return 0*X[:, 0:1]

def func_const(val):
    """
    Return an array of constant values.

    :param val: Constant value to return.
    :type val: float
    :return: Function that returns array of constant value.
    :rtype: func

    """
    def output(X):
        """
        Return an array of constant values.

       :param X: Coordinates at which zeros are found.
       :type X: numpy array
       :return: Constant matching number of points in X.
       :rtype: numpy array

        """
        return 0*X[:, 0:1] + val

    return output
############################################
def addWeights(loss_weights,weights):
    """
    Adds list of weights to the full list of weights.

    :param loss_weights: Current list of weights.
    :param weights: List of weights to append.
    :type loss_weights: list
    :type weights: list
    :return: Combined list of weights
    :rtype: list

    """
    for w in weights:
        loss_weights.append(w)

    return loss_weights
#############################################
