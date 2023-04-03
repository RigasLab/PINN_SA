import os
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

from PINNBox.PINNPlotter import plotAllLoss
from PINNBox.CaseBox.PH.SetupPH import setupPH, getPHPlot

def initCase(case, getAnchor = False, test_path = "Test_A/", other = [], pde_type = "", weights = [1], resolution = '0p5'):
    if case == "PH":
        base_path = "/Results_PH/"
        [domain, pde, bc, loss_name, loss_weights, transform, func_sol, X_anchor, aux_D] = setupPH(pde_type, other[0], other[1], other[2], getAnchor, weights, resolution)

        if (pde_type == "RST-E"):
            if (other[1][0]):
                netShape = [2,7]
            else:        
                netShape = [2,6]
        elif (pde_type == "HD"):
            if (other[1][0]):
                netShape = [2,6]
            else:        
                netShape = [2,5]
        else:
            print("PDE Type not configured")
            quit()

    else:
        # Add other cases
        print("No Case Match")
        quit()

    path_list = makeDir(base_path,test_path)

    return [domain, pde, bc, loss_name, loss_weights, netShape, transform, func_sol, X_anchor, path_list, aux_D]


def makeDir(base_path,test_path):
    path = os.getcwd()

    full_path = path + base_path + test_path

    path_pre = "Model/Pre"
    path_adam = "Model/Adam"
    path_lbfgs = "Model/LBFGS"
    path_init = "Init"

    path_M = full_path + path_pre
    path_A = full_path + path_adam
    path_B = full_path + path_lbfgs
    path_C = full_path + path_init

    print(full_path)
    try:
        os.makedirs(path_A)
    except FileExistsError:
        print ("Directory %s already exists" % path_A)
    
    try:
        os.makedirs(path_B)
    except FileExistsError:
        print ("Directory %s already exists" % path_B)
    
    try:
        os.makedirs(path_C)
    except FileExistsError:
        print ("Directory %s already exists" % path_C)

    try:
        os.makedirs(path_M)
    except FileExistsError:
        print ("Directory %s already exists" % path_M)

    return [full_path, path_M, path_A, path_B, path_C]

def postProTrain(history, path, loss_name, fname, model_name, model):
    loss_train = np.asarray(history.loss_train,dtype=np.float32)
    loss_test = np.asarray(history.loss_test,dtype=np.float32)
    loss_step = np.asarray(history.steps,dtype=np.float32)

    np.save(path + "/" + fname + "_tr.npy",loss_train)
    np.save(path + "/" + fname + "_te.npy",loss_test)
    np.save(path + "/" + fname + "_step.npy",loss_step)

    model.save(path + model_name)

    return

def getPlotter(case):
    if case == "PH":
        f = getPHPlot()

    else:
        ## Add Other cases
        print("No Case Match")
        quit()

    return f

