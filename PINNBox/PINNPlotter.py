import numpy as np
import matplotlib.pyplot as plt

def plotAllLoss(history,dir,pre="",loss_name=[]):
    loss_train = np.asarray(history.loss_train,dtype=np.float32)
    loss_test = np.asarray(history.loss_test,dtype=np.float32)

    for i,l in enumerate(loss_name):
        plotLoss(history.steps,loss_train[:,i],loss_test[:,i],l,dir,pre)

    plotLoss(history.steps,np.sum(loss_train,axis=1),np.sum(loss_test,axis=1),"All",dir,pre)

def plotLoss(iter,loss_train,loss_test,lab,dir,pre=""):
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
    plt.rc('legend', fontsize=13)    # legend fontsize
    plt.rc('lines', markersize=1)
    #plt.rcParams.update({
    #    "text.usetex": True,
    #    "font.family": "serif",
    #    "font.serif": ["Times"]})

    plt.semilogy(iter, loss_train, label="Train loss")
    plt.semilogy(iter, loss_test, label="Test loss")

    plt.xlabel("# Steps")
    plt.ylabel("Loss")
    plt.legend()

    fname = pre + "_Loss_" + lab
    plt.savefig(str(dir + fname + ".png"))
    plt.clf()
