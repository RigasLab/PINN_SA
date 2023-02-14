import csv
import numpy as np
#import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from PeriodicHills.Point import Point
from PeriodicHills.PeriodicHillPlotter import plotDel, plotAbs
from PeriodicHills.PeriodicHillFunctions import genH, getBSLHill

# This class stores data for an entire hill geometry and allows manipulations
# Inputs:
#       alpha       -       The stretch factor representing the scaling of this hill rel. to alpha=1
#       simType     -       The simulation methodology e.g. RANS, DNS etc
#       twoD        -       Whether the hill simulation is a 2D geometry
#       axis        -       If 2D, which axis is the 2D plane
#       pos         -       If 2D, which station(on axis) is it 2D about
#       turbulent   -       Do you want to keep turbulent data
class PeriodicHill:
    def __init__(
        self,
        fpath,
        ftype = "csv",
        alpha = 1.0,
        simType = "RANS",
        twoD = True, # 2D stuff in dictionary?
        axis = "z", # 2D stuff in dictionary?
        pos = 0.0, # 2D stuff in dictionary?
        turbulent = False,
        rms = "MEAN",
        dtype = np.float32
        ):
        self.dtype = dtype
        #Store hill equation parameters and calculat length of hill - in dict?
        self.l = 5.142
        self.m = 3.858
        self.Lx = self.calculateL(alpha)

        #Store alpha and simType
        self.alpha = alpha
        self.simType = simType

        # Read all point from file and store in list points
        if (ftype == "csv"):
            self.points = self.readCSV(fpath)
        elif (ftype == "dat"):
            self.points = self.readDAT(fpath,rms)

        #[x,q] = self.toNumpy()
        #self.plotOne("Plot/",str(self.alpha) + "_" + "raw",x[:,0:2],q,var=0)

        #Extract all points depending on if 2D, turbulent etc.
        self.points = self.extractData(twoD=twoD, axis=axis, pos=pos, turbulent=turbulent)
        
        #[xh,yh] = getBSLHill()

        #[x,q] = self.toNumpy()
        
        #plt.scatter(x[:,0], x[:,1],c=q[:,2],marker = ',')
        #plt.plot(xh,yh,'k')

        #plt.set_cmap('seismic')
        #plt.colorbar()

        #plt.show()
        #self.plotOne("Plot/",str(self.alpha) + "_" + "untransformed",x,q,var=0)

        #Transform data to scaled form
        self.transformData()

        #[x,q] = self.toNumpy()
        #self.plotOne("Plot/",str(self.alpha) + "_" + "transformed_u",x,q,var=0)
        #self.plotOne("Plot/",str(self.alpha) + "_" + "transformed_v",x,q,var=1)
        #self.plotOne("Plot/",str(self.alpha) + "_" + "transformed_p",x,q,var=2)

        #self.transformData()
        #self.removeUnderHill()
        #p1 = self.transformDataOld()
        #p2 = self.transformDataNew()

        ##Transform data to scaled form
        #self.transformData()

        #[x,q] = self.toNumpy()
        #self.plotOne("Plot/",str(self.alpha) + "_" + "transformed_u",x,q,var=0)
        #self.plotOne("Plot/",str(self.alpha) + "_" + "transformed_v",x,q,var=1)
        #self.plotOne("Plot/",str(self.alpha) + "_" + "transformed_p",x,q,var=2)

        #self.points = p1

        #[x,q] = self.toNumpy()
        #self.plotOne("Plot/",str(self.alpha) + "_" + "transformed_u_old",x,q,var=0)
        #self.plotOne("Plot/",str(self.alpha) + "_" + "transformed_v_old",x,q,var=1)
        #self.plotOne("Plot/",str(self.alpha) + "_" + "transformed_p_old",x,q,var=2)

        #self.points = p2

        #[x2,q2] = self.toNumpy()
        #self.plotOne("Plot/",str(self.alpha) + "_" + "transformed_u_new",x2,q2,var=0)
        #self.plotOne("Plot/",str(self.alpha) + "_" + "transformed_v_new",x2,q2,var=1)
        #self.plotOne("Plot/",str(self.alpha) + "_" + "transformed_p_new",x2,q2,var=2)

        #q_temp = griddata(x2,q2,x, method='linear')#,fill_value = 0.0)
        #err_q = np.subtract(q_temp,q)
        #err_q[np.isnan(err_q)] = 0.0

        #self.plotOne("Plot/",str(self.alpha) + "_" + "err_u",x,err_q,var=0)
        #self.plotOne("Plot/",str(self.alpha) + "_" + "err_v",x,err_q,var=1)
        #self.plotOne("Plot/",str(self.alpha) + "_" + "err_p",x,err_q,var=2)

    # This function reads CSV fname, extracts the header,
    # converts it into correct form and then stores each point in Point object
    def readCSV(self,fname):
        with open(fname) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
    
            headers = next(readCSV,None)
            headers = list(map(self.headerMap,headers))
            
            p = []
            for row in readCSV:
                p.append(Point(row,headers))
        return p

    # This function reads DAT fname, extracts the header,
    # converts it into correct form and then stores each point in Point object
    def readDAT(self,fname,rms):
        data = np.loadtxt(fname)
        if (rms == "MEAN"):
            headers = ['X','Y','UMEAN','VMEAN','WMEAN','PMEAN']
        elif (rms == "RMS1"):
            headers = ['X','Y','UUMEAN','VVMEAN','WWMEAN','PPMEAN']
            
        elif (rms == "RMS2"):
            headers = ['X','Y','UVMEAN','UWMEAN','VWMEAN']

        headers = list(map(self.headerMap,headers))

        #y correction?
        #yp = np.loadtxt(yp.dat)
        p = []
        p2 = []
        for i in range(0,data.shape[0]):
            p.append(Point(data[i,:],headers))

            if (data[i,0] == 0.0):
                temp = data[i,:]
                temp[0] = self.Lx
                p2.append(temp)

        for i in range(0,len(p2)):
            p.append(Point(p2[i],headers))


        return p

    def headerMap(self,header):
        if ((header == "Points:0") or (header == "X")):
            return "x"
        elif ((header == "Points:1") or (header == "Y")):
            return "y"
        elif ((header == "Points:2") or (header == "Z")):
            return "z"
        elif ((header == "U:0") or (header == "UMEAN")):
            return "u"
        elif ((header == "U:1") or (header == "VMEAN")):
            return "v"
        elif ((header == "U:2") or (header == "WMEAN")):
            return "w"
        elif ((header == "p") or (header == "PMEAN")):
            return "p"
        ### OPTIMISE FROM HERE ON IN
        #grad(p)
        elif ((header == "grad(p):0") or (header == "Px")):
            return "dp_dx"
        elif ((header == "grad(p):1") or (header == "Py")):
            return "dp_dy"
        elif (header == "grad(p):2"):
            return "dp_dz"
        #grad(U)
        elif ((header == "grad(U):0")  or (header == "grad(UMEAN):0") or (header == "Ux")):
            return "du_dx"
        elif ((header == "grad(U):1") or (header == "grad(UMEAN):1") or (header == "Uy")):
            return "du_dy"
        elif ((header == "grad(U):2") or (header == "grad(UMEAN):2")):
            return "du_dz"
        elif ((header == "grad(U):3") or (header == "Vx")):
            return "dv_dx"
        elif ((header == "grad(U):4") or (header == "Vy")):
            return "dv_dy"
        elif (header == "grad(U):5"):
            return "dv_dz"
        elif (header == "grad(U):6"):
            return "dw_dx"
        elif (header == "grad(U):7"):
            return "dw_dy"
        elif (header == "grad(U):8"):
            return "dw_dz"
        #grad2(U)
        elif ((header == "grad(grad(U)xx):0") or (header == "Uxx")):
            return "d2u_dx2"
        elif ((header == "grad(grad(U)xy):1") or (header == "Uyy")):
            return "d2u_dy2"
        elif ((header == "grad(grad(U)yx):0") or (header == "Vxx")):
            return "d2v_dx2"
        elif ((header == "grad(grad(U)yy):1") or (header == "Vyy")):
            return "d2v_dy2"
        #Reynolds Stress
        elif (header == "turbulenceProperties:R:0"):
            return "tau_xx"
        elif (header == "turbulenceProperties:R:1"):
            return "tau_xy"#yy
        elif (header == "turbulenceProperties:R:2"):
            return "tau_xz"#zz
        elif (header == "turbulenceProperties:R:3"):
            return "tau_yy"#xy
        elif (header == "turbulenceProperties:R:4"):
            return "tau_yz"#yz
        elif (header == "turbulenceProperties:R:5"):
            return "tau_zz"#xz
        #grad(tau)
        elif ((header == "grad(turbulenceProperties:Rxx):0") or (header == "uux")):
            return "duu_dx"
        elif ((header == "grad(turbulenceProperties:Rxy):0") or (header == "uvx")):
            return "duv_dx"
        elif ((header == "grad(turbulenceProperties:Rxy):1") or (header == "uvy")):
            return "duv_dy"
        elif ((header == "grad(turbulenceProperties:Ryy):1") or (header == "vvy")):
            return "dvv_dy"
        #tau
        elif (header == "UUMEAN"):
            return "uu"
        elif (header == "VVMEAN"):
            return "vv"
        elif (header == "WWMEAN"):
            return "ww"
        elif (header == "UVMEAN"):
            return "uv"
        elif (header == "UWMEAN"):
            return "uw"
        elif (header == "VWMEAN"):
            return "vw"
        else:
            return header

    def extractData(self,twoD=True,axis="z",pos=0.0,turbulent = False):
        p = []

        for point in self.points:
            if (not turbulent):
                point.removeTurbulence()

            if (twoD):
                if (point.is2D(axis = axis, pos = pos)):
                    point.extract2D(axis)
                    p.append(point)
            else:
                p.append(point)

        return p

    def normalise(self, velNorm=1, presNorm=1):
        for point in self.points:
            point.normalise(velNorm=velNorm,presNorm=presNorm)

    def calculateL(self,alpha):
        return self.m*alpha + self.l

    def transformData(self):
        p = []
        xMax = 0
        for point in self.points:
            x = self.rescaleHills(point.X["x"])
            x = self.recentreData(x)
            #if (x > xMax):
            #    xMax = x
            #NEED TO RECTIFY 'PROPERLY'
            point.X["x"] = x
            p.append(point)

        #if (xMax < 4.5):
        #    p2 = []
        #    for point in p:
        #        if (point.X["x"] == xMax):
        #            point.X["x"] = 4.5
        #        #NEED TO RECTIFY 'PROPERLY'
        #        p2.append(point)
        #    p = p2

        self.points = p

    def transformDataInvert(self):
        p = []
        xMax = 0
        for point in self.points:
            x = point.X["x"]
            if (x > xMax):
                xMax = x
         
        xMax2 = 0
        for point in self.points:
            xN = -(point.X["x"]-xMax)
            x = self.rescaleHills(xN)
            x = self.recentreData(x)
            if (x > xMax2):
                xMax2 = x
            #NEED TO RECTIFY 'PROPERLY'
            point.X["x"] = x
            p.append(point)

        if (xMax2 < 4.5):
            p2 = []
            for point in p:
                if (point.X["x"] == xMax2):
                    point.X["x"] = 4.5
                #NEED TO RECTIFY 'PROPERLY'
                p2.append(point)
            p = p2

        self.points = p

    def rescaleHills(self, x):
        #a = self.alpha
        #Lx = self.calculateL(a)
        W = 1.929*self.alpha
        xi = 0

        if (x <= W) :
            xi = (1/self.alpha)*x
        elif (x >=  self.Lx-W):
            xi = (1/self.alpha)*x + ((1/self.alpha)-1)*(2*W-self.Lx)
        else:
            xi = x + W*((1/self.alpha)-1)

        return xi

    def recentreData(self,x):
        return x - 4.5

    def removeUnderHill(self):
        p = []
        count = 0
        for point in self.points:
            x = point.X["x"]
            y = point.X["y"]

            if (self.isUnder(x + 4.5,y)):
                #print("UNDER")
                count += 1
            else:
                p.append(point)
        self.points = p

    def isUnder(self,x,y):
        h = genH(x)

        if (y < h):
            return True
        else:
            return False

    def toNumpy(self):
        x = []
        q = []
        #more efficient to use vstack?
        for point in self.points:
            x.append(point.xToNumpy())
            q.append(point.qToNumpy())

        x = np.asarray(x, dtype=self.dtype)
        q = np.asarray(q, dtype=self.dtype)

        return [x,q]

    def getVolume(self):
        vol = []
        #more efficient to use vstack?
        for point in self.points:
            vol.append(point.getVol())

        vol = np.asarray(vol, dtype=self.dtype)

        return vol

    def getAlpha(self):
        return self.alpha

      # NEEDS TESTING
    def calcGrad(self,dX):
        #ask each hill to calculate grads

        [X,Q] = self.toNumpy()

        n = X.shape[1]
        Xmin = np.zeros(n)
        Xmax = np.zeros(n)

        for i in range(0,n):
            Xmin[i] = np.amin(X[:,i])
            Xmax[i] = np.amax(X[:,i])

        X_p = np.zeros((n,X.shape[0],n))
        X_m = np.zeros((n,X.shape[0],n))

        for i in range(0,n):
            X_p[i] = X
            X_p[i,:,i] += dX[i]

            X_m[i] = X
            X_m[i,:,i] -= dX[i]

        Q_p = np.zeros((n,X.shape[0],Q.shape[1]))
        Q_m = np.zeros((n,X.shape[0],Q.shape[1]))

        for i in range(0,n):
            Q_p[i] = griddata(X,Q,X_p[i], method='linear')#,fill_value = 0.0)
            Q_m[i] = griddata(X,Q,X_m[i], method='linear')#,fill_value = 0.0)

        X_over = []
        X_under = []
        X_IX = []

        for i in range(0,n):
            X_over.append(X_p[i,:,i] > Xmax[i])
            X_under.append(X_m[i,:,i] < Xmin[i])
            X_IX.append(~(X_over[i] + X_under[i]))

        dQdX = np.zeros((n,Q.shape[0],Q.shape[1]))

        for i in range(0,n):
            dQdX[i,X_IX[i],:] = (Q_p[i,X_IX[i],:] - Q_m[i,X_IX[i],:])/(2*dX[i])
            dQdX[i,X_over[i],:] = (Q[X_over[i],:] - Q_m[i,X_over[i],:])/(dX[i])
            dQdX[i,X_under[i],:] = (Q_p[i,X_under[i],:] - Q[X_under[i],:])/(dX[i])

        ind = np.isnan(dQdX)
        dQdX[ind] = 0.0

        return dQdX

    # NEEDS TESTING
    def addGrad(self,dX):
        qKeys = self.points[0].getQKeys()
        xKeys = self.points[0].getXKeys()

        headers = []

        for x in xKeys:
            for q in qKeys:
                headers.append("d" + q + "_d" + x)

        dQdX = self.calcGrad(dX)
        [X,Q] = self.toNumpy()

        tQ = dQdX[0]
        for i in range(1,dQdX.shape[0]):
            tQ = np.hstack((tQ,dQdX[i]))

        dQdX = tQ
        for i,p in enumerate(self.points):
            self.points[i].addToQ(headers,dQdX[i,:])

    def plotAll(self,dir,fname):
        [X,Q] = self.toNumpy()
        qKeys = self.points[0].getQKeys()

        for i in range(0,Q.shape[1]):
            fname2 = fname+str("_")+qKeys[i]
            self.plotOne(dir,fname2,X,Q[:,i])

    def plotOne(self,dir,fname,X,Q,var=0):
        #plotAbs(X,Q,dir,fname,var)
        plotDel(X,Q,dir,fname,var)

