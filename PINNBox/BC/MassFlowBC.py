import numbers
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np
import tensorflow as tf

from deepxde import backend as bkd
from deepxde import config
from deepxde import utils
from deepxde.backend import backend_name

class MassFlowBC(object):
    """Dirichlet boundary condition for a set of points.
    Compare the output (that associates with `points`) with `values` (target data).

    Args:
        points: An array of points where the corresponding target values are known and used for training.
        values: An array of values that gives the exact solution of the problem.
        component: The output component satisfying this BC.
    """

    def __init__(self, points, values, component=0, dy = []):
        self.points = np.array(points, dtype=config.real(np))
        if not isinstance(values, numbers.Number) and values.shape[1] != 1:
            raise RuntimeError(
                "PointSetBC should output 1D values. Use argument 'component' for different components."
            )
        self.values = bkd.as_tensor(values, dtype=config.real(bkd.lib))
        self.component = component
        self.Lin = 2.036
        self.dy = dy

    def collocation_points(self, X):
        return self.points

    def error(self, X, inputs, outputs, beg, end):
        Uin = outputs[beg:end,0:1]
        
        integral = tf.math.reduce_sum(Uin*self.dy)/self.Lin

        return integral - 1.0