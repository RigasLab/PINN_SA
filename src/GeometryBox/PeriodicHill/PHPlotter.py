import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
#####################################################################################################################
#Plots all flow fields in variables from var
def plotFieldDataset(X,Q,alpha,var = [0],dir = "",fname = ""):
    for i in range(0,alpha.shape[0]):
        for v in var:
            if (v==0):
                suf = "_u"
            elif (v==1):
                suf = "_v"
            elif (v==2):
                suf = "_p"

            plotDel(X[i],Q[i],dir,str(fname + str(alpha[i])+suf),v)

#Plots all predictions in variables from var - plots prediction, truth and error
def plotComparisonDataset(X,Qpred,Qtruth,alpha,var = [0],dir = "",fname = "", doPlot=[True,True,True]):
    for i in range(0,alpha.shape[0]):
        for v in var:
            if (v==0):
                suf = "_u"
            elif (v==1):
                suf = "_v"
            elif (v==2):
                suf = "_p"

            plotComparison(X[i],Qpred[i],Qtruth[i],dir=dir,fname=str(fname + str(alpha[i])+suf),var=v,doPlot=doPlot)

#Plots all predictions in variables from var - plots prediction, truth and error
def plotErrorComparisonDataset(X,Qtruth,alpha,err1=[],err2=[],Q1=[],Q2=[],calcErr=False,label_list=["1","2"],var = [0],dir = "",fname = "",plotTruth=False):
    if(calcErr):      
        err1 = Q1-Qtruth
        err2 = Q2-Qtruth

    for i in range(0,alpha.shape[0]):
        for v in var:
            if (v==0):
                suf = "_u"
            elif (v==1):
                suf = "_v"
            elif (v==2):
                suf = "_p"

            plotErrorComparison(X[i],Qtruth[i],err1[i],err2[i],label_list=label_list,dir=dir,fname=str(fname + str(alpha[i])+suf),var=v,plotTruth=plotTruth)
#######################################################################################################
def plotComparison(X,Qpred,Qtruth,image=False,lim=(),dir="Plot/",fname="",var=0, doPlot=[True,True,True]):
    Qpred = Qpred[:,var]
    Qtruth = Qtruth[:,var]

    err = Qpred-Qtruth

    vmax = np.amax(np.abs(Qtruth))
    vmax2 = np.amax(np.abs(Qpred))
    if (vmax2 > vmax):
        vmax = vmax2

    #vmax = 0.02

    if (doPlot[0]):
        plotter(X,Qtruth,dir=dir,fname=str(fname + "_Truth"),type="spec_scaled",vmax=vmax,image=image,lim=lim,xlab="$x$",ylab="$y$")
    if (doPlot[1]):
        plotter(X,Qpred,dir=dir,fname=str(fname + "_Predict"),type="spec_scaled",vmax=vmax,image=image,lim=lim,xlab="$x$",ylab="$y$")
    if (doPlot[2]):
        plotter(X,err,dir=dir,fname=str(fname + "_Err"),type="scaled",image=image,lim=lim,xlab="$x$",ylab="$y$")

def plotErrorComparison(X,Qtruth,err1,err2,image=False,lim=(),label_list=["1","2"],dir="Plot/",fname="",var=0,plotTruth=False):
    Qtruth = Qtruth[:,var]
    err1 = err1[:,var]
    err2 = err2[:,var]

    vmax = np.amax(np.abs(err1))
    vmax2 = np.amax(np.abs(err2))
    if (vmax2 > vmax):
        vmax = vmax2

    plotter(X,err1,dir=dir,fname=str(fname + "_Err_"+label_list[0]),type="spec_scaled",vmax=vmax,image=image,lim=lim,xlab="$x$",ylab="$y$")
    plotter(X,err2,dir=dir,fname=str(fname + "_Err_"+label_list[1]),type="spec_scaled",vmax=vmax,image=image,lim=lim,xlab="$x$",ylab="$y$")
    if (plotTruth):
        plotter(X,Qtruth,dir=dir,fname=str(fname + "_Truth"),type="scaled",image=image,lim=lim,xlab="$x$",ylab="$y$")

def plotErrorTrainOverlay(X,Qpred,Qtruth,Xoverlay,image=False,lim=(),dir="Plot/",fname="",var=0):
    Qpred = Qpred[:,var]
    Qtruth = Qtruth[:,var]

    err = Qpred-Qtruth

    plotter(X,err,dir=dir,fname=str(fname + "_Err_Overlay"),type="scaled",image=image,lim=lim,overlay=True,Xoverlay=Xoverlay,xlab="$x$",ylab="$y$")

def plotAbs(X,Q,dir="Plot/",fname="",var=0):
    plotter(X,Q[:,var],dir=dir,fname=fname,type="basic",xlab="x",ylab="y")

def plotPoints(X,dir,fname):
    plotter(X,dir=dir,fname=fname,type="points",xlab="x",ylab="y")

def plotDel(X,Q,dir="Plot/",fname="",var=0):
    plotter(X,Q[:,var],dir=dir,fname=fname,type="scaled",xlab="x",ylab="y")

def plotDelErr(X,Qpred,Qtruth,dir="Plot/",fname="",var=0):
    err = Qpred[:,var]-Qtruth[:,var]
    plotter(X,err,dir=dir,fname=str(fname + "_Err"),type="scaled",xlab="x",ylab="y")

def plotR(X,R,dir,fname,var):
    plotter(X,R[:,var],dir=dir,fname=fname,type="R",xlab="x",ylab="y")
##########################################################################################################
def plotter(X=[],Q=[],dir="Plot/",fname="",type="scaled",vmax=0,image=False,lim=(),overlay=False,Xoverlay=[],xlab="$x$",ylab="$y$"):
    if(type=="scaled"):
        val = np.abs(Q)
        vmax = np.amax(val)

        if(image):
            plotImage(Q,lim=lim,vmin=-vmax,vmax=vmax,dir=dir,fname=fname,xlab=xlab,ylab=ylab)
        else:
            if(overlay):
                #vmax = 0.05
                plotScatterOverlay(X,Q,Xoverlay,vmin=-vmax,vmax=vmax,dir=dir,fname=fname,xlab=xlab,ylab=ylab)
            else:
                plotScatter(X,Q,vmin=-vmax,vmax=vmax,dir=dir,fname=fname,xlab=xlab,ylab=ylab)
        
    elif(type=="spec_scaled"):
        if(image):
            plotImage(Q,lim=lim,vmin=-vmax,vmax=vmax,dir=dir,fname=fname,xlab=xlab,ylab=ylab)
        else:
            if(overlay):
                plotScatterOverlay(X,Q,Xoverlay,vmin=-vmax,vmax=vmax,dir=dir,fname=fname,xlab=xlab,ylab=ylab)
            else:
                plotScatter(X,Q,vmin=-vmax,vmax=vmax,dir=dir,fname=fname,xlab=xlab,ylab=ylab)

    elif(type=="R"):
        if(image):
            plotImage(Q,lim=lim,vmin=-1,vmax=1,dir=dir,fname=fname,xlab=xlab,ylab=ylab)
        else:
            plotScatter(X,Q,vmin=-1,vmax=1,dir=dir,fname=fname,xlab=xlab,ylab=ylab)

    elif(type=="basic"):
        if(image):
            plotImage(Q,cmap='viridis',lim=lim,scale=False,dir=dir,fname=fname,xlab=xlab,ylab=ylab)
        else:
            plotScatter(X,Q,cmap='viridis',scale=False,dir=dir,fname=fname,xlab=xlab,ylab=ylab)

    elif(type=="points"):
        plotPoints(X,dir=dir,fname=fname,xlab=xlab,ylab=ylab)

def plotScatter(X,Q,cmap="seismic",scale=True,vmin=0,vmax=0,dir="Plot/",fname="",xlab="$x$",ylab="$y$"):
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=19)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=17)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=17)    # fontsize of the tick labels
    plt.rc('legend', fontsize=13)    # legend fontsize
    #plt.rc('lines', markersize=0.005)
    plt.rc('lines', markersize=1)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"]})

    if (scale):
        plt.scatter(X[:,0], X[:,1],c=Q,marker = ',',vmin=vmin, vmax=vmax)
    else:
        plt.scatter(X[:,0], X[:,1],c=Q,marker = ',')

    plt.set_cmap(cmap)
    plt.colorbar()

    plt.ylabel(ylab)
    plt.xlabel(xlab)

    #plt.savefig(str(dir + fname + ".pdf"))
    #plt.savefig(str(dir + fname + ".eps"))
    plt.savefig(str(dir + fname + ".png"))
    plt.clf()

def plotScatterOverlay(X,Q,Xoverlay,cmap="seismic",scale=True,vmin=0,vmax=0,dir="Plot/",fname="",xlab="$x$",ylab="$y$"):
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=19)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=17)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=17)    # fontsize of the tick labels
    plt.rc('legend', fontsize=13)    # legend fontsize
    #plt.rc('lines', markersize=0.005)
    plt.rc('lines', markersize=1)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"]})

    if (scale):
        plt.scatter(X[:,0], X[:,1],c=Q,marker = ',',vmin=vmin, vmax=vmax)
    else:
        plt.scatter(X[:,0], X[:,1],c=Q,marker = ',')

    plt.set_cmap(cmap)
    plt.colorbar()

    plt.plot(Xoverlay[:,0], Xoverlay[:,1],'go',markersize=4)

    plt.ylabel(ylab)
    plt.xlabel(xlab)

    #plt.savefig(str(dir + fname + ".pdf"))
    #plt.savefig(str(dir + fname + ".eps"))
    plt.savefig(str(dir + fname + ".png"))
    plt.clf()

def plotPoints(X,dir="Plot/",fname="",xlab="x",ylab="y"):
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
    plt.rc('legend', fontsize=13)    # legend fontsize
    plt.rc('lines', markersize=1)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"]})

    plt.plot(X[:,0], X[:,1])
    
    plt.ylabel(ylab)
    plt.xlabel(xlab)

    plt.savefig(str(dir + fname + ".png"))
    plt.clf()

def plotImage(Q,lim=(),cmap="seismic",scale=True,vmin=0,vmax=0,axis_equal=False,dir="Plot/",fname="",xlab="x",ylab="y"):
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
    plt.rc('legend', fontsize=13)    # legend fontsize
    plt.rc('lines', markersize=1)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"]})

    if (scale):
        plt.imshow(Q[:,:],vmin=vmin,vmax=vmax,origin='lower',extent=lim)
    else:
        plt.imshow(Q[:,:],origin='lower',extent=lim)
    
    plt.set_cmap(cmap)
    plt.colorbar()

    if(axis_equal):
        plt.axis('equal')

    plt.ylabel(ylab)
    plt.xlabel(xlab)

    plt.savefig(str(dir + fname + ".png"))
    plt.clf()
##########################################################################################################
def plotMultiProfile(xp,x_shift,sf,xh,yh,X=[],Q=[],linestyle_list=[],legend_lab=[]):
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=15)    # legend fontsize
    plt.rc('lines', markersize=1)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"]})
    X_in = np.array([[0,1],[0,3.036]])
    X_out = np.array([[9,1],[9,3.036]])
    X_top = np.array([[0,3.036],[9,3.036]])

    xp -= x_shift
    X_in[:,0] -= x_shift
    X_out[:,0] -= x_shift
    X_top[:,0] -= x_shift

    ax = plt.gca()
    ax.plot(xh,yh,'k-')
    ax.plot(X_in[:,0],X_in[:,1],'k-')
    ax.plot(X_out[:,0],X_out[:,1],'k-')
    ax.plot(X_top[:,0],X_top[:,1],'k-')

    for x_t in xp:
        line_list = []
        for i in range(0,len(X)):
            [X_Profile,Q_Profile] = getProfile(X[i],Q[i],x_t)

            line, = ax.plot(X_Profile[:,0]+sf*Q_Profile[:,0],X_Profile[:,1],linestyle_list[i])
            line_list.append(line)

        X_BSL = np.array([[x_t,0],[x_t,3.036]])
        ax.plot(X_BSL[:,0],X_BSL[:,1],'k--')

    #plt.axis('equal')
    ax.legend(line_list, legend_lab, loc='lower right')
    plt.ylabel('$\eta$')
    plt.xlabel('$\\xi$')
    plt.show()

def getProfile(X,Q,xp):
    #Y_Sample = np.linspace(0,3.036,500)
    #X_Sample = xp*np.ones(Y_Sample.shape)

    #X_Profile = np.column_stack((X_Sample,Y_Sample))

    #Q_Profile = griddata(X,Q,X_Profile,method='linear')
    dx = np.square(X[:,0]-xp)

    IX = dx.argmin()

    tol = 1e-10
    ind = (X[:,0] == X[IX,0])

    X_Profile = X[ind,:]
    sortX = np.argsort(X_Profile[:,1])

    Q_Profile = Q[ind,:]

    X_Profile = X_Profile[sortX,:]
    Q_Profile = Q_Profile[sortX,:]

    return [X_Profile,Q_Profile]
################################################################################################
def genH(x):
    if (x > 4.5):
        x = 4.5 - (x - 4.5)

    hill = 0
    if (x >= 0) and (x <= 0.3214):
        x *= 28
        hill = 1 + (0)*x + (2.420e-4)*(x*x) - (7.588e-5)*(x*x*x)
        hill = min(1,hill)
    elif (x > 0.3214) and (x <= 0.5):
        x *= 28
        hill = 0.8955 + (3.484e-2)*x - (3.629e-3)*(x*x) + (6.749e-5)*(x*x*x)
    elif (x > 0.5) and (x <= 0.7143):
        x *= 28
        hill = 0.9213 + (2.931e-2)*x - (3.234e-3)*(x*x) + (5.809e-5)*(x*x*x)
    elif (x > 0.7143) and (x <= 1.071):
        x *= 28
        hill = 1.445 - (4.927e-2)*x + (6.950e-4)*(x*x) - (7.394e-6)*(x*x*x)
    elif (x > 1.071) and (x <= 1.429):
        x *= 28
        hill = 0.6401 + (3.123e-2)*x - (1.988e-3)*(x*x) + (2.242e-5)*(x*x*x)
    elif (x > 1.429) and (x <= 1.929):
        x *= 28
        hill = 2.0139 - (7.180e-2)*x + (5.875e-4)*(x*x) + (9.553e-7)*(x*x*x)
        hill = max(0,hill)
    else:
        hill = 0

    return hill

def getBSLHill():
    xh = np.linspace(0,9,1000)
    yh = np.zeros(1000)

    for i,x in enumerate(xh):
        yh[i] = genH(x)
    return [xh,yh]
################################################################################################
##########################################################################################################
def uniformGrid(X,Q,nX,nY, interp = 'linear', fill = 0.0):
    xmin = min(X[:,0])
    xmax = max(X[:,0])
    ymin = min(X[:,1])
    ymax = max(X[:,1])

    x = np.linspace(xmin, xmax, nX)
    y = np.linspace(ymin, ymax, nY)

    A, B = np.meshgrid(x, y)
    X0 = np.column_stack((A.flatten(),B.flatten()))

    Q0 = griddata(X,Q,X0, method=interp, fill_value = fill)
    #Q1 = griddata(X,Q,(A,B), method=interp, fill_value = fill)

    return [X0,Q0]#,Q1,A,B]

def plot2Points(X1,X2):
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
    plt.rc('legend', fontsize=13)    # legend fontsize

    plt.plot(X1[:,0], X1[:,1],'bo')
    plt.plot(X2[:,0], X2[:,1],'rx')

    plt.show()
    plt.clf()























####################################################################################################
#def plotter(X,Q,dir="Plot/",fname="",type="scaled",xlab="x",ylab="y"):
#    plt.rc('axes', titlesize=16)     # fontsize of the axes title
#    plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
#    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
#    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
#    plt.rc('legend', fontsize=13)    # legend fontsize
#    plt.rc('lines', markersize=1)

#    if(type=="scaled"):
#        val = np.abs(Q)
#        vmax = np.amax(val)
#        #vmax = 0.6
#        plt.scatter(X[:,0], X[:,1],c=Q,marker = ',',vmin = -vmax, vmax = vmax)

#        plt.set_cmap('seismic')
#        plt.colorbar()

#    elif(type=="R"):
#        plt.scatter(X[:,0], X[:,1],c=Q,marker = ',',vmin = -1, vmax = 1)

#        plt.set_cmap('seismic')
#        plt.colorbar()

#    elif(type=="basic"):
#        plt.scatter(X[:,0], X[:,1],c=Q,marker = ',')
#        #plt.pcolormesh(A, B, Q1)
#        plt.set_cmap('viridis')
#        plt.colorbar()

#    elif(type=="points"):
#        plt.plot(X[:,0], X[:,1])

#    plt.ylabel(ylab)
#    plt.xlabel(xlab)
#    plt.savefig(str(dir + fname + ".png"))
#    #plt.show()
#    plt.clf()

#def plotter2(X,Q,dir="Plot/",fname="",xlab="x",ylab="y",vmax=1):
#    plt.rc('axes', titlesize=16)     # fontsize of the axes title
#    plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
#    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
#    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
#    plt.rc('legend', fontsize=13)    # legend fontsize
#    plt.rc('lines', markersize=1)

#    plt.scatter(X[:,0], X[:,1],c=Q,marker = ',',vmin = -vmax, vmax = vmax)

#    plt.set_cmap('seismic')
#    plt.colorbar()

#    plt.ylabel(ylab)
#    plt.xlabel(xlab)
#    plt.savefig(str(dir + fname + ".png"))
#    #plt.show()
#    plt.clf()

#def plotAbs(X,Q,dir="Plot/",fname="",var=0):
#    #[X,Q] = uniformGrid(X,Q[:,var],250,250)#1000,1000 #500,500
#    Q = Q[:,var]
#    plotter(X,Q,dir=dir,fname=fname,type="basic",xlab="x",ylab="y")


#def plotPoints(X,dir,fname):
#    plotter(X,X[:,0],dir=dir,fname=fname,type="points",xlab="x",ylab="y")


#def plotDel(X,Q,dir="Plot/",fname="",var=0):
#    #[X,Q] = uniformGrid(X,Q[:,var],250,250)
#    Q = Q[:,var]
#    plotter(X,Q,dir=dir,fname=fname,type="scaled",xlab="x",ylab="y")


#def plotDelErr(X,Qpred,Qtruth,dir="Plot/",fname="",var=0):
#    Qpred = Qpred[:,var]
#    Qtruth = Qtruth[:,var]

#    err = Qpred-Qtruth
    
#    plotter(X,err,dir=dir,fname=str(fname + "_Err"),type="scaled",xlab="x",ylab="y")

#def plotR(X,R,dir,fname,var):
#    #[X,R] = uniformGrid(X,R[:,var],250,250)
#    R = R[:,var]
#    plotter(X,R,dir=dir,fname=fname,type="R",xlab="x",ylab="y")

#def imPlotter(Q,dir="Plot/",fname="",xlab="x",ylab="y",lim=()):
#    plt.rc('axes', titlesize=16)     # fontsize of the axes title
#    plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
#    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
#    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
#    plt.rc('legend', fontsize=13)    # legend fontsize

#    val = np.abs(Q)
#    vmax = np.amax(val)

#    plt.imshow(Q[:,:],vmin=-vmax,vmax=vmax,origin='lower',extent=lim)
    
#    plt.set_cmap('seismic')
#    plt.colorbar()

#    #plt.axis('equal')
#    plt.ylabel(ylab)
#    plt.xlabel(xlab)

#    plt.savefig(str(dir + fname + ".png"))
#    #plt.show()
#    plt.clf()


#def imPlotter2(Q,dir="Plot/",fname="",xlab="x",ylab="y",vmax=1,lim=()):
#    plt.rc('axes', titlesize=16)     # fontsize of the axes title
#    plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
#    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
#    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
#    plt.rc('legend', fontsize=13)    # legend fontsize

#    plt.imshow(Q[:,:],vmin=-vmax,vmax=vmax,origin='lower',extent=lim)
    
#    plt.set_cmap('seismic')
#    plt.colorbar()

#    #plt.axis('equal')
#    #plt.axis('square')
#    plt.ylabel(ylab)
#    plt.xlabel(xlab)

#    plt.savefig(str(dir + fname + ".png"))
#    #plt.show()
#    plt.clf()

#def plotDelComp2(Qpred,Qtruth,lim,dir="Plot/",fname="",var=0):
#    Qpred = Qpred[:,:,var]
#    Qtruth = Qtruth[:,:,var]

#    err = (Qpred-Qtruth)#/Qtruth

#    vmax = np.amax(np.amax(Qtruth))
#    vmax2 = np.amax(np.amax(Qpred))

#    if (vmax2 > vmax):
#        vmax = vmax2

#    imPlotter2(Qtruth,dir=dir,fname=str(fname + "_Truth"),xlab="x",ylab="y",vmax=vmax,lim=lim)
#    #imPlotter2(Qpred,dir=dir,fname=str(fname + "_Predict"),xlab="x",ylab="y",vmax=vmax,lim=lim)
#    #imPlotter(err,dir=dir,fname=str(fname + "_Err"),xlab="x",ylab="y",lim=lim)
############################################################################################################
#def plotDelComp3(X,Qpred,Qtruth,dir="Plot/",fname=""):
#    Qpred = Qpred[:]
#    Qtruth = Qtruth[:]

#    err = Qpred-Qtruth

#    vmax = np.amax(np.amax(Qtruth))
#    vmax2 = np.amax(np.amax(Qpred))
#    if (vmax2 > vmax):
#        vmax = vmax2
    
#    plotter2(X,Qtruth,dir=dir,fname=str(fname + "_Truth"),xlab="x",ylab="y",vmax=vmax)
#    plotter2(X,Qpred,dir=dir,fname=str(fname + "_Predict"),xlab="x",ylab="y",vmax=vmax)
#    plotter(X,err,dir=dir,fname=str(fname + "_Err"),type="scaled",xlab="x",ylab="y")

#def plotComp(X,Qpred,Qtruth,dir="Plot/",fname="",var=0):
#    Qpred = Qpred[:]
#    Qtruth = Qtruth[:]

#    err = Qpred-Qtruth

#    vmax = np.amax(np.amax(Qtruth))
#    vmax2 = np.amax(np.amax(Qpred))
#    if (vmax2 > vmax):
#        vmax = vmax2
    
#    plotter(X,Qtruth,dir=dir,fname=str(fname + "_Truth"),type="basic",xlab="x",ylab="y")#,vmax=vmax)
#    plotter(X,Qpred,dir=dir,fname=str(fname + "_Predict"),type="basic",xlab="x",ylab="y")#,vmax=vmax)
#    plotter(X,err,dir=dir,fname=str(fname + "_Err"),type="scaled",xlab="x",ylab="y")
