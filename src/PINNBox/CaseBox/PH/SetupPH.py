import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

from deepxde.backend import tf

from PINNBox.PDEBox.RANS import RANS_B_RSTE,RANS_B_HD,RANS_SA_HD, getSATerms

from PINNBox.BaseFunctions import func_zeros, func_const, addWeights
from PeriodicHills.PeriodicHillPlotter import plotComparison

from PeriodicHills.PeriodicHillDataset import PeriodicHillDataset
from PeriodicHills.PeriodicHillDDEGeometryBase import PeriodicHillDDEGeometryBase

from PeriodicHills.PeriodicHillBC import boundary_wall, boundary_outlet
from PINNBox.BC.MassFlowBC import MassFlowBC

from PeriodicHills.PeriodicHillPINNFunctions import loadQ0
from PeriodicHills.PeriodicHillFunctions import genHArr, getD

from scipy.interpolate import griddata
import os
import csv

doSA = False
X_test = []

#[domain, pde, bc, loss_name, loss_weights, netShape, transform, func_sol, X_anchor, path_list]
def setupPH(pde_type="HD",fpList=[True,-0.0110],SA=[True, 1e-10], massflow=[False,1000],getAnchor=False,weights=[1],resolution='0p5'):
    rho = 1.0
    Re = 5600
    U = 1.0
    P = 0.0
    H = 1.0
    nu = U*H/Re

    fixedfp = fpList[0]
    fp = fpList[1]

    global doSA
    doSA = SA[0]
    minS = SA[1] 

    domdot = massflow[0]
    Nmdot  = massflow[1]

    domain = PeriodicHillDDEGeometryBase()
    ##############################
    if (doSA):
        if (pde_type == "RST-E"):
            print("RST-E with SA not configured")
            quit()   
        elif (pde_type == "HD"):          
            pde = RANS_SA_HD(rho,nu,minS,fp)
    else:
        if (pde_type == "RST-E"):
            pde = RANS_B_RSTE(rho,nu,fp)
        elif (pde_type == "HD"):          
            pde = RANS_B_HD(rho,nu,fp)
    #############################################################
    removeUpper = True

    refineUpper = {"refine":False,
                   "res":10,
                   "x0":0.0}
    #############################################################

    global X_test
    [_,_,_,X_test,Q_test,tau_test] = loadQ0("InputData/DNS/XDE",10,5,refineUpper,removeUpper)
    
    bc = genBC(domain, pde_type, doSA, domdot, Nmdot, [X_test,Q_test,tau_test], resolution)

    [loss_name,loss_weights] = genLosses(pde_type, doSA, domdot, weights)

    transform = h_c(pde_type)

    func_sol = sol(X_test, Q_test)

    aux_D = func_D()

    if (getAnchor):
        X_anchor = np.load("******")

    else:
        X_anchor = None

    return [domain, pde, bc, loss_name, loss_weights, transform, func_sol, X_anchor, aux_D]

#############################################################################
def sol(X_test,Q_test):
    def output(X):
        sol = griddata(X_test,Q_test,X, method='linear', fill_value = 0.0)
        return sol
    return output 

def func_D():
    return getD

def h_c(pde_type):
    if(pde_type == "HD"):
        def output(X,Q):
            U, V, P, T, fsu, fsv = Q[:, 0:1], Q[:, 1:2], Q[:, 2:3], Q[:, 3:4], Q[:, 4:5], Q[:, 5:6]
            T_new = T**2
            return tf.concat((U, V, P, T_new, fsu, fsv), axis=1)
    elif(pde_type == "RST-E"):
        def output(X,Q):
            U, V, P, T, uu, uv, vv = Q[:, 0:1], Q[:, 1:2], Q[:, 2:3], Q[:, 3:4], Q[:, 4:5], Q[:, 5:6], Q[:, 6:7]
            T_new = T**2
            return tf.concat((U, V, P, T_new, uu, uv, vv), axis=1)

    return output
#########################################################################
def getTrainingData(X_test, Q_test, tau_test, resolution):
    if(resolution == '0p3'):
       nX = "31"
       nY = "11"
    elif(resolution == '0p4'):
       nX = "24"
       nY = "9"
    elif(resolution == '0p5'):
       nX = "19"
       nY = "7"
    elif(resolution == '0p6'):
       nX = "16"
       nY = "6"
    elif(resolution == '1p0'):
       nX = "10"
       nY = "4"
    elif(resolution == '0p2'):
       nX = "46"
       nY = "16"
    elif(resolution == '0p1'):
       nX = "91"
       nY = "31"
    elif(resolution == '0p05'):
       nX = "181"
       nY = "61"
    elif(resolution == '0p01'):
       nX = "901"
       nY = "301"
    elif(resolution == '0p8_b'):
       resolution = "0p8"
       nX = "12"
       nY = "4_bottom"
    elif(resolution == '0p8_c'):
       resolution = "0p8"
       nX = "12"
       nY = "4_centre"
    elif(resolution == '0p8_t'):
       resolution = "0p8"
       nX = "12"
       nY = "4_top"
    else:
        resolution = '0p5'
        nX = "19"
        nY = "7"

    measure_fname = "InputData/DNS/OM/assimilated_sol_" + resolution + "/" + "measurements_nx_" + nX + "_ny_" + nY + ".txt"

    X_measure = np.loadtxt(measure_fname, skiprows=1)
    X_measure[:,0] -= 4.5

    h = genHArr(X_measure[:,0] + 4.5)

    IX = (X_measure[:,1] > h[:])

    X_pts = X_measure[IX,:]
    
    Q_pts     = griddata(X_test,Q_test,X_pts, method='linear', fill_value = 0.0)
    tau_pts   = griddata(X_test,tau_test,X_pts, method='linear', fill_value = 0.0)

    return [X_pts, Q_pts, tau_pts]

def genBC(domain, pde_type, doSA, domdot, Nmdot, datList, resolution):

    [X_pts, Q_pts, tau_pts] = getTrainingData(datList[0],datList[1],datList[2], resolution)
    ###################################################
    if (pde_type == "HD"):
        bc = genBC_HD(domain, doSA, domdot, Nmdot, X_pts, Q_pts)
    elif (pde_type == "RST-E"):
        bc = genBC_RST_E(domain, doSA, domdot, Nmdot, X_pts, Q_pts, tau_pts)
    
    return bc

def genBC_HD(domain, doSA, doMass, N, X_pts, Q_pts):
    ############################################################################
    #Data
    U_points = dde.icbc.PointSetBC(X_pts,Q_pts[:,0:1], component=0)
    V_points = dde.icbc.PointSetBC(X_pts,Q_pts[:,1:2], component=1)
    ###########################
    if(doMass):
        #Massflow
        Lin = 3.036-1.0
        dy = Lin/N
        yinit = np.linspace(1,3.036,N+1)

        Yin = 0.5*(yinit[1:] + yinit[:-1])

        X_in = np.zeros((N,2))
        X_in[:,0] = -4.5
        X_in[:,1] = Yin
        bc_massflow      = MassFlowBC(X_in,X_in[:,0:1], component=0, dy = dy)
    ###########################
    # Wall
    bc_wall_U       = dde.icbc.DirichletBC(domain,func_zeros,boundary_wall,component=0)
    bc_wall_V       = dde.icbc.DirichletBC(domain,func_zeros,boundary_wall,component=1)
    
    if (doSA):
        bc_wall_T       = dde.icbc.DirichletBC(domain,func_zeros,boundary_wall,component=3)
        bc_wall_fsu     = dde.icbc.DirichletBC(domain,func_zeros,boundary_wall,component=4)
        bc_wall_fsv     = dde.icbc.DirichletBC(domain,func_zeros,boundary_wall,component=5)
    else:
        bc_wall_fsu     = dde.icbc.DirichletBC(domain,func_zeros,boundary_wall,component=3)
        bc_wall_fsv     = dde.icbc.DirichletBC(domain,func_zeros,boundary_wall,component=4)
    ###########################
    #Periodic
    bc_periodic_U   = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=0)
    bc_periodic_V   = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=1)
    bc_periodic_P   = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=2)
    
    if (doSA):
        bc_periodic_T   = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=3)
        bc_periodic_fsu = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=4)
        bc_periodic_fsv = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=5)
    else:
        bc_periodic_fsu = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=3)
        bc_periodic_fsv = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=4)

    if (doSA):
        bc = [U_points, V_points,
              bc_wall_U, bc_wall_V, bc_wall_T, bc_wall_fsu, bc_wall_fsv, 
              bc_periodic_U, bc_periodic_V, bc_periodic_P, bc_periodic_T, bc_periodic_fsu, bc_periodic_fsv]
    else:
        bc = [U_points, V_points,
              bc_wall_U, bc_wall_V, bc_wall_fsu, bc_wall_fsv, 
              bc_periodic_U, bc_periodic_V, bc_periodic_P, bc_periodic_fsu, bc_periodic_fsv]

    if(doMass):
        bc.append(bc_massflow)

    return bc

def genBC_RST_E(domain, doSA, doMass, N, X_pts, Q_pts, tau_pts):
    ############################################################################
    #Data
    U_points = dde.icbc.PointSetBC(X_pts,Q_pts[:,0:1], component=0)
    V_points = dde.icbc.PointSetBC(X_pts,Q_pts[:,1:2], component=1)

    if (doSA):
        uu_points = dde.icbc.PointSetBC(X_pts,tau_pts[:,0:1], component=4)
        uv_points = dde.icbc.PointSetBC(X_pts,tau_pts[:,1:2], component=5)
        vv_points = dde.icbc.PointSetBC(X_pts,tau_pts[:,2:3], component=6)
    else:
        uu_points = dde.icbc.PointSetBC(X_pts,tau_pts[:,0:1], component=3)
        uv_points = dde.icbc.PointSetBC(X_pts,tau_pts[:,1:2], component=4)
        vv_points = dde.icbc.PointSetBC(X_pts,tau_pts[:,2:3], component=5)
    ###########################
    if(doMass):
        #Massflow
        Lin = 3.036-1.0
        dy = Lin/N
        yinit = np.linspace(1,3.036,N+1)

        Yin = 0.5*(yinit[1:] + yinit[:-1])

        X_in = np.zeros((N,2))
        X_in[:,0] = -4.5
        X_in[:,1] = Yin
        bc_massflow      = MassFlowBC(X_in,X_in[:,0:1], component=0, dy = dy)
    ###########################
    # Wall
    bc_wall_U       = dde.icbc.DirichletBC(domain,func_zeros,boundary_wall,component=0)
    bc_wall_V       = dde.icbc.DirichletBC(domain,func_zeros,boundary_wall,component=1)
    
    if (doSA):
        bc_wall_T       = dde.icbc.DirichletBC(domain,func_zeros,boundary_wall,component=3)
        bc_wall_uu     = dde.icbc.DirichletBC(domain,func_zeros,boundary_wall,component=4)
        bc_wall_uv     = dde.icbc.DirichletBC(domain,func_zeros,boundary_wall,component=5)
        bc_wall_vv     = dde.icbc.DirichletBC(domain,func_zeros,boundary_wall,component=6)
    else:
        bc_wall_uu     = dde.icbc.DirichletBC(domain,func_zeros,boundary_wall,component=3)
        bc_wall_uv     = dde.icbc.DirichletBC(domain,func_zeros,boundary_wall,component=4)
        bc_wall_vv     = dde.icbc.DirichletBC(domain,func_zeros,boundary_wall,component=5)
    ###########################
    #Periodic
    bc_periodic_U   = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=0)
    bc_periodic_V   = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=1)
    bc_periodic_P   = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=2)
    
    if (doSA):
        bc_periodic_T   = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=3)
        bc_periodic_uu = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=4)
        bc_periodic_uv = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=5)
        bc_periodic_vv = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=6)
    else:
        bc_periodic_uu = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=3)
        bc_periodic_uv = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=4)
        bc_periodic_vv = dde.icbc.PeriodicBC(domain, 0, boundary_outlet, component=5)

    if (doSA):
        bc = [U_points, V_points, uu_points, uv_points, vv_points,
              bc_wall_U, bc_wall_V, bc_wall_T, bc_wall_uu, bc_wall_uv, bc_wall_vv, 
              bc_periodic_U, bc_periodic_V, bc_periodic_P, bc_periodic_T, bc_periodic_uu, bc_periodic_uv, bc_periodic_vv]
    else:
        bc = [U_points, V_points, uu_points, uv_points, vv_points,
              bc_wall_U, bc_wall_V, bc_wall_uu, bc_wall_uv, bc_wall_vv, 
              bc_periodic_U, bc_periodic_V, bc_periodic_P, bc_periodic_uu, bc_periodic_uv, bc_periodic_vv]

    if(doMass):
        bc.append(bc_massflow)

    return bc


def genLosses(pde_type,doSA,domdot,weights):
    if (pde_type == "HD"):
        [loss_name,loss_weights] = genLosses_HD(doSA,domdot,weights)
    elif (pde_type == "RST-E"):
        [loss_name,loss_weights] = genLosses_RST_E(doSA,domdot,weights)

    return [loss_name,loss_weights]

def genLosses_HD(doSA,doMass,weights):
    weight_pde      = weights[0]
    weight_reg      = weights[1]
    weight_data     = weights[2]
    weight_wall     = weights[3]
    weight_periodic = weights[4]
    weight_massflow = weights[5]

    #########################################
    ### Loss Weights ###
    loss_weights = []
    
    loss_weights = addWeights(loss_weights,weight_pde)
    if (doSA):
        loss_weights = addWeights(loss_weights,weight_reg) 
    #############
    loss_weights = addWeights(loss_weights,weight_data)
    loss_weights = addWeights(loss_weights,weight_wall)
    loss_weights = addWeights(loss_weights,weight_periodic)
    if(doMass):
        loss_weights = addWeights(loss_weights,weight_massflow)
    #########################################
    print("Number of Losses: " + str(len(loss_weights)))
    print("Number of PDEs: " + str(len(weight_pde)))
    print("Number of Other(BC) Losses: " + str(len(loss_weights) - len(weight_pde)))
    ##########################################################################
    ### Loss Name ###
    loss_name = []

    loss_name.append('mass')
    loss_name.append('x-mom')
    loss_name.append('y-mom')
    loss_name.append('div(fs)')
    if(doSA):
        loss_name.append('SA')
        loss_name.append('minF')
    #################
    loss_name.append('BC_data_U')
    loss_name.append('BC_data_V')
    #################
    loss_name.append('BC_wall_U')
    loss_name.append('BC_wall_V')
    if(doSA):
        loss_name.append('BC_wall_T')
    loss_name.append('BC_wall_fsu')
    loss_name.append('BC_wall_fsv')
    #################
    loss_name.append('BC_periodic_U')
    loss_name.append('BC_periodic_V')
    loss_name.append('BC_periodic_P')
    if(doSA):
        loss_name.append('BC_periodic_T')
    loss_name.append('BC_periodic_fsu')
    loss_name.append('BC_periodic_fsv')
    if(doMass):
        loss_name.append('BC_massflow')

    return [loss_name, loss_weights]
##########################################################################
def genLosses_RST_E(doSA,doMass,weights):
    weight_pde      = weights[0]
    weight_reg      = weights[1]
    weight_data     = weights[2]
    weight_wall     = weights[3]
    weight_periodic = weights[4]
    weight_massflow = weights[5]

    #########################################
    ### Loss Weights ###
    loss_weights = []
    
    loss_weights = addWeights(loss_weights,weight_pde)
    if (doSA):
        loss_weights = addWeights(loss_weights,weight_reg) 
    #############
    loss_weights = addWeights(loss_weights,weight_data)
    loss_weights = addWeights(loss_weights,weight_wall)
    loss_weights = addWeights(loss_weights,weight_periodic)
    if(doMass):
        loss_weights = addWeights(loss_weights,weight_massflow)
    #########################################
    print("Number of Losses: " + str(len(loss_weights)))
    print("Number of PDEs: " + str(len(weight_pde)))
    print("Number of Other(BC) Losses: " + str(len(loss_weights) - len(weight_pde)))
    ##########################################################################
    ### Loss Name ###
    loss_name = []

    loss_name.append('mass')
    loss_name.append('x-mom')
    loss_name.append('y-mom')
    if(doSA):
        loss_name.append('SA')
        loss_name.append('minF')
    #################
    loss_name.append('BC_data_U')
    loss_name.append('BC_data_V')
    loss_name.append('BC_data_uu')
    loss_name.append('BC_data_uv')
    loss_name.append('BC_data_vv')
    #################
    loss_name.append('BC_wall_U')
    loss_name.append('BC_wall_V')
    if(doSA):
        loss_name.append('BC_wall_T')
    loss_name.append('BC_wall_uu')
    loss_name.append('BC_wall_uv')
    loss_name.append('BC_wall_vv')
    #################
    loss_name.append('BC_periodic_U')
    loss_name.append('BC_periodic_V')
    loss_name.append('BC_periodic_P')
    if(doSA):
        loss_name.append('BC_periodic_T')
    loss_name.append('BC_periodic_uu')
    loss_name.append('BC_periodic_uv')
    loss_name.append('BC_periodic_vv')
    if(doMass):
        loss_name.append('BC_massflow')

    return [loss_name, loss_weights]
##########################################################################
def getPHPlot():
    def processResults(plot_dir,func_sol,model,pde):
        #U,V,P,T,fsu,fsv
        #Residuals
        #Sp,Sdiff,Sc,Sdestr?    

        if (doSA):
            getSA = getSATerms(1.0,1/5600,1e-4)
            other_func = [pde,getSA]

        else:
            other_func = [pde]         
    
        plotter(X_test,model,func_sol,plot_dir,"",other=other_func)
        ##############################################################
        return

    return processResults

def plotter(X,model,func_sol,dir,suf,other):
    Q_pred = model.predict(X)
    Q_truth = func_sol(X)

    #plotComparison(X,Q_pred,Q_truth,dir=dir,fname="U"+suf,var=0, doPlot=[False,True,False])
    #plotComparison(X,Q_pred,Q_truth,dir=dir,fname="V"+suf,var=1, doPlot=[False,True,False])
    #plotComparison(X,Q_pred,Q_truth,dir=dir,fname="P"+suf,var=2, doPlot=[False,True,False])
    #plotComparison(X,Q_pred,Q_truth,dir=dir,fname="nutilde"+suf,var=3, doPlot=[False,True,False])

    np.save(dir+"X" + suf + ".npy",X)
    np.save(dir+"Q_Pred" + suf + ".npy",Q_pred)
    np.save(dir+"Q_Truth" + suf + ".npy",Q_truth)
    #########################################################################################
    res_err = model.predict(X, operator = other[0])

    res_err = np.asarray(res_err,dtype=np.float32)
    res_err = np.squeeze(res_err, axis=2)
    res_err = np.swapaxes(res_err, 0,1)

    #plotComparison(X,res_err,np.zeros(res_err.shape),dir=dir,fname="Mass_Residual"+suf,var=0, doPlot=[False,True,False])
    #plotComparison(X,res_err,np.zeros(res_err.shape),dir=dir,fname="MomX_Residual"+suf,var=1, doPlot=[False,True,False])
    #plotComparison(X,res_err,np.zeros(res_err.shape),dir=dir,fname="MomY_Residual"+suf,var=2, doPlot=[False,True,False])
    #plotComparison(X,res_err,np.zeros(res_err.shape),dir=dir,fname="SA_Residual"+suf,var=3, doPlot=[False,True,False])

    np.save(dir+"Residuals" + suf + ".npy",res_err)
    #########################################################################################
    #########################################################################################
    if(doSA):
        SA_terms = model.predict(X, operator = other[1])

        SA_terms = np.asarray(SA_terms,dtype=np.float32)
        SA_terms = np.squeeze(SA_terms, axis=2)
        SA_terms = np.swapaxes(SA_terms, 0,1)

        #plotComparison(X,res_err,np.zeros(res_err.shape),dir=dir,fname="Sp"+suf,var=0, doPlot=[False,True,False])
        #plotComparison(X,res_err,np.zeros(res_err.shape),dir=dir,fname="Sd"+suf,var=1, doPlot=[False,True,False])
        #plotComparison(X,res_err,np.zeros(res_err.shape),dir=dir,fname="Sdiff"+suf,var=2, doPlot=[False,True,False])
        #plotComparison(X,res_err,np.zeros(res_err.shape),dir=dir,fname="Sc"+suf,var=3, doPlot=[False,True,False])

        np.save(dir+"SA_terms" + suf + ".npy",SA_terms)

    return
###################################################

       

     

