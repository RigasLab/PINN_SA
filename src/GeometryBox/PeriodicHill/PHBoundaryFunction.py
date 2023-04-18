import numpy as np

from GeometryBox.PeriodicHill.PHAuxFunctions import genH
####################################################################################################################
####################################################################################################################
def boundary_wall(x, on_boundary):
    return on_boundary and (
        np.isclose(x[1],3.036)
        or checkHill(x)
        )

def boundary_wall_2(x, on_boundary):
    return on_boundary and (
        boundary_wall_upper(x)
        or boundary_wall_lower(x)
        )

def boundary_wall_upper(x):
    return np.isclose(x[1],3.036)

def boundary_wall_lower(x):
    return np.isclose(x[1],genH(x[0]+4.5))

def checkHill(x):
    return np.isclose(x[1],genH(x[0]+4.5))
####################################################################################################################
def boundary_periodic(x, on_boundary):
    return on_boundary and (
        np.isclose(x[0],-4.5)
        or np.isclose(x[0],4.5)
        )

def boundary_periodic_2(x, on_boundary):
    return on_boundary and boundary_periodic_outlet(x)

def boundary_outlet(x, on_boundary):
    return on_boundary and np.isclose(x[0],4.5)

def boundary_periodic_outlet(x):
    return np.isclose(x[0],4.5) and np.greater_equal(x[1], 1.0)

def boundary_periodic_inlet(x):
    return np.isclose(x[0],-4.5) and np.greater_equal(x[1], 1.0)
####################################################################################################################
