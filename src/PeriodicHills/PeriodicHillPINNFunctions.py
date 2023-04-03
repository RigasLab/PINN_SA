#import deepxde as dde
import numpy as np

import tensorflow as tf

from scipy.interpolate import griddata

from PeriodicHills.PeriodicHillFunctions import genHArr, genH, getHillSegment

from PeriodicHills.PeriodicHillDataset import PeriodicHillDataset
####################################################################################################################
####################################################################################################################
##TODO REFACTOR
def loadQ0(dir,nX,nY,refineUpper={"refine":False},removeUpper=False,dtype=np.float32):
    print("LOADING QMEAN")
    dataset0 = PeriodicHillDataset(dir = dir + "/mean", simType = "DNS", ftype = "dat",turbulent=True, rms="MEAN", dtype=dtype)
    print(dataset0.hills[0].points[0].getQKeys())

    [modelDict,testDict, valDict, shuffleDict] = dataset0.getTrainingDict()

    valDict["validate"] = False
    testDict["alpha"] = [0.0]
    modelDict["alpha0"] = 1.0
    modelDict["remHill"] = True

    modelDict["nX"] = nX
    modelDict["nY"] = nY

    modelDict["normalX"] = "none"
    modelDict["normaldQ"] = "none"

    modelDict["single"] = True
    modelDict["difference"] = "none"

    #################################################################################################################
    modelDict["grid"] = "rectangular"
    [X0,Q0,_,_,_,_] = dataset0.getData(modelDict=modelDict,testDict=testDict,valDict=valDict,shuffleDict=shuffleDict)

    N = X0[0].shape[0]
    Xn = np.zeros((N,2))
    Qn = np.zeros((N,3))
    taun = np.zeros((N,6))

    Xn[:,0] = X0[0]
    Xn[:,1] = X0[1]

    Qn[:,0] = Q0[0]
    Qn[:,1] = Q0[1]
    Qn[:,2] = Q0[2]

    modelDict["grid"] = "base"
    [X0,Q0,_,_,_,_] = dataset0.getData(modelDict=modelDict,testDict=testDict,valDict=valDict,shuffleDict=shuffleDict)

    N = X0[0].shape[0]
    Xt = np.zeros((N,2))
    Qt = np.zeros((N,3))
    taut = np.zeros((N,6))

    Xt[:,0] = X0[0]
    Xt[:,1] = X0[1]

    Qt[:,0] = Q0[0]
    Qt[:,1] = Q0[1]
    Qt[:,2] = Q0[2]

    #################################################################################################################
    print("LOADING QRMS1")
    dataset0 = PeriodicHillDataset(dir = dir + "/rms1", simType = "DNS", ftype = "dat",turbulent=True, rms="RMS1",dtype=dtype)
    print(dataset0.hills[0].points[0].getQKeys())

    modelDict["grid"] = "rectangular"
    [_,Q0,_,_,_,_] = dataset0.getData(modelDict=modelDict,testDict=testDict,valDict=valDict,shuffleDict=shuffleDict)

    taun[:,0] = Q0[0]
    taun[:,2] = Q0[1]
    taun[:,5] = Q0[2]

    modelDict["grid"] = "base"
    [_,Q0,_,_,_,_] = dataset0.getData(modelDict=modelDict,testDict=testDict,valDict=valDict,shuffleDict=shuffleDict)

    taut[:,0] = Q0[0]
    taut[:,2] = Q0[1]
    taut[:,5] = Q0[2]

    #################################################################################################################
    print("LOADING QRMS2")
    dataset0 = PeriodicHillDataset(dir = dir + "/rms2", simType = "DNS", ftype = "dat",turbulent=True, rms="RMS2",dtype=dtype)
    print(dataset0.hills[0].points[0].getQKeys())

    modelDict["grid"] = "rectangular"
    [_,Q0,_,_,_,_] = dataset0.getData(modelDict=modelDict,testDict=testDict,valDict=valDict,shuffleDict=shuffleDict)

    taun[:,1] = Q0[0]
    taun[:,3] = Q0[1]
    taun[:,4] = Q0[2]

    modelDict["grid"] = "base"
    [_,Q0,_,_,_,_] = dataset0.getData(modelDict=modelDict,testDict=testDict,valDict=valDict,shuffleDict=shuffleDict)

    taut[:,1] = Q0[0]
    taut[:,3] = Q0[1]
    taut[:,4] = Q0[2]

    #################################################################################################################
    if(refineUpper["refine"]):
        [Xn,Qn,taun] = addUpperBoundaryResolution(refineUpper["res"],nY,refineUpper["x0"],
                                                  Xt,Qt,taut,
                                                  Xn,Qn,taun)
    if(removeUpper):
        [Xn,Qn,taun] = removeUpperData(Xn,Qn,taun)
    #################################################################################################################

    [Xn,Qn,taun] = removeUnderHill(Xn,Qn,taun)

    return [Xn,Qn,taun,Xt,Qt,taut]

    ##[u,v,p,uux,vvy,uvx,uvy,Ux,Uy,Uxx,Uyy,Vx,Vy,Vxx,Vyy,Px,Py]
####################################################################################################################
####################################################################################################################
def getWallPressure(X,P,nT,nB,dT,dB):
    Xb = np.zeros((nB,2))
    Xt = np.zeros((nT,2))

    xb = np.linspace(0,9,nB)
    h = genHArr(xb)

    Xb[:,0] = xb - 4.5
    Xb[:,1] = h + dB

    xt = np.linspace(0,9,nT)
    h = 3.036

    Xt[:,0] = xt - 4.5
    Xt[:,1] = h - dT

    Pb = griddata(X,P,Xb, method='linear', fill_value = 0.0)
    Pt = griddata(X,P,Xt, method='linear', fill_value = 0.0)

    X_n = np.append(Xb,Xt,axis=0)
    P_n = np.append(Pb,Pt,axis=0)

    return [X_n,P_n]
####################################################################################################################
def getInletPressure(X,P,nY,dX):
    Xn = np.zeros((nY,2))

    h = genH(dX)
    y = np.linspace(h,3.036,nY)

    Xn[:,0] = dX - 4.5
    Xn[:,1] = y

    Pn = griddata(X,P,Xn, method='linear', fill_value = 0.0)

    return [Xn,Pn]
#####################################################################################################################
####################################################################################################################
def addUpperBoundaryResolution(res,nY,x0,X_full,Q_full,tau_full,X,Q,tau):
    dy = 3.036/nY

    y_pl = np.linspace(0,dy,res)

    X_pl = np.zeros((res,2))
    X_pl[:,0] = x0
    X_pl[:,1] = 3.036 - y_pl

    Q_pl = griddata(X_full,Q_full,X_pl, method='linear', fill_value = 0.0)
    tau_pl = griddata(X_full,tau_full,X_pl, method='linear', fill_value = 0.0)

    X_n = np.append(X,X_pl,axis=0)
    Q_n = np.append(Q,Q_pl,axis=0)
    tau_n = np.append(tau,tau_pl,axis=0)

    return [X_n,Q_n,tau_n]
####################################################################################################################
####################################################################################################################
def removeUpperData(X,Q,tau):

    IX = np.isclose(X[:,1],3.036)
    IX = (IX == False)

    X_n = X[IX,:]
    Q_n = Q[IX,:]
    tau_n = tau[IX,:]

    return [X_n,Q_n,tau_n]
####################################################################################################################
####################################################################################################################
def removeUnderHill(X,Q,tau):
    h = genHArr(X[:,0]+4.5)

    IX = (X[:,1] > h[:])

    X_n = X[IX,:]
    Q_n = Q[IX,:]
    tau_n = tau[IX,:]

    return [X_n,Q_n,tau_n]
####################################################################################################################
####################################################################################################################
def loadQ(dir,nX,nY,refineUpper={"refine":False},removeUpper=False,alpha_test=0.0):
    print("LOADING QMEAN")
    dataset0 = PeriodicHillDataset(dir = dir + "/mean", simType = "DNS", ftype = "dat",turbulent=True, rms="MEAN")
    print(dataset0.hills[0].points[0].getQKeys())

    [modelDict,testDict, valDict, shuffleDict] = dataset0.getTrainingDict()

    valDict["validate"] = False
    testDict["alpha"] = [alpha_test]
    modelDict["alpha0"] = 1.0
    modelDict["remHill"] = True

    modelDict["nX"] = nX
    modelDict["nY"] = nY

    modelDict["normalX"] = "none"
    modelDict["normaldQ"] = "none"

    modelDict["single"] = True
    modelDict["difference"] = "none"

    #################################################################################################################
    modelDict["grid"] = "rectangular"
    [X0,Q0,Alpha0,_,_,_] = dataset0.getData(modelDict=modelDict,testDict=testDict,valDict=valDict,shuffleDict=shuffleDict)

    N = X0[0].shape[0]
    Xn = np.zeros((N,3))
    Qn = np.zeros((N,3))
    taun = np.zeros((N,6))

    Xn[:,0] = X0[0]
    Xn[:,1] = X0[1]
    Xn[:,2] = Alpha0

    Qn[:,0] = Q0[0]
    Qn[:,1] = Q0[1]
    Qn[:,2] = Q0[2]

    testDict["alpha"] = [0.0]
    modelDict["grid"] = "base"
    [X0,Q0,Alpha0,_,_,_] = dataset0.getData(modelDict=modelDict,testDict=testDict,valDict=valDict,shuffleDict=shuffleDict)

    N = X0[0].shape[0]
    Xt = np.zeros((N,3))
    Qt = np.zeros((N,3))
    taut = np.zeros((N,6))

    Xt[:,0] = X0[0]
    Xt[:,1] = X0[1]
    Xt[:,2] = Alpha0

    Qt[:,0] = Q0[0]
    Qt[:,1] = Q0[1]
    Qt[:,2] = Q0[2]

    #################################################################################################################
    print("LOADING QRMS1")
    dataset0 = PeriodicHillDataset(dir = dir + "/rms1", simType = "DNS", ftype = "dat",turbulent=True, rms="RMS1")
    print(dataset0.hills[0].points[0].getQKeys())

    testDict["alpha"] = [alpha_test]
    modelDict["grid"] = "rectangular"
    [_,Q0,_,_,_,_] = dataset0.getData(modelDict=modelDict,testDict=testDict,valDict=valDict,shuffleDict=shuffleDict)

    taun[:,0] = Q0[0]
    taun[:,2] = Q0[1]
    taun[:,5] = Q0[2]

    testDict["alpha"] = [0.0]
    modelDict["grid"] = "base"
    [_,Q0,_,_,_,_] = dataset0.getData(modelDict=modelDict,testDict=testDict,valDict=valDict,shuffleDict=shuffleDict)

    taut[:,0] = Q0[0]
    taut[:,2] = Q0[1]
    taut[:,5] = Q0[2]

    #################################################################################################################
    print("LOADING QRMS2")
    dataset0 = PeriodicHillDataset(dir = dir + "/rms2", simType = "DNS", ftype = "dat",turbulent=True, rms="RMS2")
    print(dataset0.hills[0].points[0].getQKeys())

    testDict["alpha"] = [alpha_test]
    modelDict["grid"] = "rectangular"
    [_,Q0,_,_,_,_] = dataset0.getData(modelDict=modelDict,testDict=testDict,valDict=valDict,shuffleDict=shuffleDict)

    taun[:,1] = Q0[0]
    taun[:,3] = Q0[1]
    taun[:,4] = Q0[2]

    testDict["alpha"] = [0.0]
    modelDict["grid"] = "base"
    [_,Q0,_,_,_,_] = dataset0.getData(modelDict=modelDict,testDict=testDict,valDict=valDict,shuffleDict=shuffleDict)

    taut[:,1] = Q0[0]
    taut[:,3] = Q0[1]
    taut[:,4] = Q0[2]

    #################################################################################################################
    if(refineUpper["refine"]):
        [Xn,Qn,taun] = addUpperBoundaryResolution(refineUpper["res"],nY,refineUpper["x0"],
                                                  Xt,Qt,taut,
                                                  Xn,Qn,taun)
    if(removeUpper):
        [Xn,Qn,taun] = removeUpperData(Xn,Qn,taun)
    #################################################################################################################

    [Xn,Qn,taun] = removeUnderHill(Xn,Qn,taun)

    return [Xn,Qn,taun,Xt,Qt,taut]

    ##[u,v,p,uux,vvy,uvx,uvy,Ux,Uy,Uxx,Uyy,Vx,Vy,Vxx,Vyy,Px,Py]
