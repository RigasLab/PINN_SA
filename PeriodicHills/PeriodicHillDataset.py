import os
from scipy.interpolate import griddata
from scipy.stats import linregress
from random import randint, uniform
import numpy as np
from math import ceil

from PeriodicHills.PeriodicHill import PeriodicHill
from PeriodicHills.PeriodicHillPlotter import plot2Points
from PeriodicHills.PeriodicHillFunctions import genH

class PeriodicHillDataset:
    def __init__(
        self,
        dir,
        simType = "RANS",
        ftype = "csv",
        twoD = True,
        axis = "z",
        pos = 0.0,
        turbulent = False,
        rms="MEAN",
        dtype = np.float32
        ):

        self.dtype = dtype
        self.hills = []
        self.numHills = 0

        self.alpha = []

        self.X = []
        self.Q = []
        self.dQ = []

        self.norm = []

        self.numInputs = 0
        self.inShape = []
        self.inName = []
        self.outShape = []
        self.outName = []

        if (ftype == "csv"):
            f_ext = ".csv"
        elif (ftype == "dat"):
            f_ext = ".dat"
        else:
            f_ext = ".txt"

        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(f_ext):
                    a = self.extractAlpha(file,ftype)
                    self.hills.append(PeriodicHill(os.path.join(root, file),ftype=ftype,alpha=a,simType=simType,twoD=twoD,axis=axis,pos=pos,turbulent=turbulent,rms=rms))
                    self.alpha.append(a)
                    self.numHills += 1

        self.alpha = np.asarray(self.alpha, dtype = self.dtype)
        self.alpha = self.alpha[:,]

    ##############################################################################
    #EXTRACT ALPHA FROM FNAME OR FIND ALPHA IN DATASET
    ##############################################################################
    def extractAlpha(self,fname,ftype):
        alpha = fname

        if (ftype == "csv"):
            f_ext = ".csv"
        elif (ftype == "dat"):
            f_ext = ".dat"
        else:
            f_ext = ".txt"

        k = alpha.find(f_ext)
        alpha = alpha[:k]
        alpha = alpha.replace("p",".")
        return float(alpha)

        #optimise
    def findAlpha(self,alpha):
        count = 0
        for a in self.alpha:
            if (a == alpha):
                return count
            count += 1

        return "ERR"

    def getHillFromAlpha(self,alpha,getX=True,getQ=True):
        ind = self.findAlpha(alpha)

        hill = []
        if (getX):
            hill.append(self.X[ind])

        if (getQ):
            hill.append(self.Q[ind])

        return hill
    ##############################################################################
    #TURN TO NUMPY
    ##############################################################################
    def toNumpy(self):
        self.X = []
        self.Q = []
        for hill in self.hills:
            [x,q] = hill.toNumpy()
            self.X.append(x)
            self.Q.append(q)

        self.X = np.asarray(self.X, dtype=self.dtype)
        self.Q = np.asarray(self.Q, dtype=self.dtype)
    ##############################################################################
    #CALCULATE SENSITIVITIES IN DATA
    ##############################################################################
    def generateDelta(self,method,alpha0=1.0):
        #change to include alpha???
        if (method == "fixed"):
            #self.toNumpy()
            return self.generateFixedDifference(alpha0,interp='linear')#'cubic')
        elif (method == "backward"):
            self.toNumpy()
            return self.generateBackwardDifference()
        elif (method == "none"):
            ind = self.findAlpha(alpha0)
            [X0,Q0] = self.hills[ind].toNumpy()

            Q = np.zeros((self.numHills,Q0.shape[0],Q0.shape[1]))
            X = np.zeros((self.numHills,X0.shape[0],X0.shape[1]))

            for i in range(0,self.numHills):
                [x,q] = self.hills[i].toNumpy()
                interp = "cubic"#"linear"
                Q[i] = griddata(x,q,X0, method=interp, fill_value = 0.0)
                X[i] = X0
            
            self.X = X
            self.Q = Q
            return self.generateAbsoluteVariables()
        else:
            print("DIFFERENCING ERROR")
            return("ERR/EXCEPTION")
    
    def generateFixedDifference(self,alpha0=1.0,interp='linear',fill=0.0):
        ind = self.findAlpha(alpha0)
        [X0,Q0] = self.hills[ind].toNumpy()  
        #[X0,Q0] = self.getHillFromAlpha(alpha0,True,True)

        deltaQ = np.zeros((self.numHills,Q0.shape[0],Q0.shape[1]))
        X = np.zeros((self.numHills,X0.shape[0],X0.shape[1]))

        for i in range(0,self.numHills):
            [x,q] = self.hills[i].toNumpy()
            #plot2Points(X0,x)
            deltaQ[i] = self.calculateDelta(x,X0,q,Q0,interp,fill)
            #deltaQ[i] = self.calculateDelta(self.X[i],X0,self.Q[i],Q0,interp,fill)
            X[i] = X0

        return [X, deltaQ]
    
    def generateBackwardDifference(self):
        [ind1,ind2] = self.findBDIndex()
        
        [Xn,Qn] = self.regrid()

        deltaQ = []
        X = []
        for i in range(0,self.numHills-1):
            deltaQ.append(self.calculateDelta(Xn[ind2[i]],Xn[ind1[i]],Qn[ind2[i]],Qn[ind1[i]]))
            X.append(Xn[ind1[i]])

        deltaQ = np.asarray(deltaQ, dtype=self.dtype)
        X = np.asarray(X, dtype=self.dtype)

        return [X, deltaQ]

    def generateAbsoluteVariables(self):
        return [self.X, self.Q]

    def findBDIndex(self):
        ind1 = []
        ind2 = []

        minAlpha = np.amin(self.alpha)
        ind1.append(self.findAlpha(minAlpha))

        for i in range(0,self.numHills-1):
            diff = 10
            count = 0
            IX = 0
            for a in self.alpha:
                diff2 = a - minAlpha
                if (diff2 > 0):
                    if ((diff2 < diff) & (diff2 != 0)):
                        diff = diff2
                        IX = count

                count += 1

            ind2.append(IX)
            if (i != self.numHills-2):
                ind1.append(IX)

            minAlpha = self.alpha[ind2[i]]

        return [ind1,ind2]
    ##############################################################################
    #REMOVE UNDERHILL REGION
    ##############################################################################
    def removeUnderHill(self,X,dQ,remHill = False):
        if (remHill == True): 
            return self.filterUnderHill(X,dQ)
        else:
            return [X,dQ]

    def filterUnderHill(self,X,dQ):
        X_new = np.zeros(X.shape)
        dQ_new = np.zeros(dQ.shape)

        count = 0
        for i in range(0,X.shape[1]):
            x = X[0,i,0]
            y = X[0,i,1]

            under = self.isUnder(x,y)

            if (under == False):
                X_new[:,count,:] = X[:,i,:]
                dQ_new[:,count,:] = dQ[:,i,:]
                count += 1

        X_new = X_new[:,0:count,:]
        dQ_new = dQ_new[:,0:count,:]

        return [X_new,dQ_new]
    
    def isUnder(self,x,y):
        h = genH(x + 4.5)

        if (y < h):
            return True
        else:
            return False
    ##############################################################################
    #SUBSET DATA
    ##############################################################################
    def getSubset(self,X,dQ,subset=False,xMin=float('-inf'),xMax=float('inf'),yMin=float('-inf'),yMax=float('inf')):
        if (subset == False):
            return [X,dQ]
        else:
            return self.calcSubset(X,dQ,xMin,xMax,yMin,yMax) 

    def calcSubset(self,X,dQ,xMin=float('-inf'),xMax=float('inf'),yMin=float('-inf'),yMax=float('inf')):
        ind = ((X[0,:,0] >= xMin) & (X[0,:,0] <= xMax) & (X[0,:,1] >= yMin) & (X[0,:,1] <= yMax))

        X = X[:,ind,:]
        dQ = dQ[:,ind,:]

        return [X,dQ]
    ##############################################################################
    #NORMALISE DATA
    ##############################################################################
    #NEED TO DEFINE MORE ROBUSTLY
    def normalise(self,Y,method="abs"):
        if (method == "none"):
            return Y
        elif (method == "abs"):
            return self.normaliseAbs(Y)
        elif (method == "normal"):
            return self.normaliseNormal(Y)
        else:
            print("NORMALISATION ERROR")
            return("ERR/EXCEPTION")

    def normaliseAbs(self,Y):
        for i in range(0,Y.shape[2]):
            norm = np.amax(np.abs(Y[:,:,i]))
            self.norm.append(norm)
            Y[:,:,i] /= norm

        return Y

    def normaliseNormal(self,Y):
        for i in range(0,Y.shape[2]):
            mu = np.mean(Y[:,:,i])
            var = np.var(Y[:,:,i])

            Y[:,:,i] = (1/np.sqrt(var))*(Y[:,:,i]-mu)
        return Y
    ##############################################################################
    #CHANGE GRID e.g Convolution , rectancgular etc.
    ##############################################################################
    def changeGrid(self,X,dQ,type,nX,nY,alpha0):
        if (type == "base"):
            return [X,dQ]
        elif (type == "rectangular"):
            return self.rectGrid(X,dQ,nX,nY,alpha0) 
        elif (type == "image"):
            return self.imGrid(X,dQ,nX,nY,alpha0) 
        else:
            print("GRID ERROR")
            return("ERR/EXCEPTION")

    def rectGrid(self,X,dQ,nX,nY,alpha0):
        ind = self.findAlpha(alpha0)
        X0 = X[ind]

        [X0,lim] = self.toMeshGrid(X0,nX,nY)

        [X_n,dQ_n] = self.interpFromMeshGrid(X0,X,dQ)

        return [X_n,dQ_n]

    def imGrid(self,X,dQ,nX,nY,alpha0):
        [X_n,dQ_n] = self.rectGrid(X,dQ,nX,nY,alpha0)

        X_n = self.toImGrid(X_n,nX,nY)
        dQ_n = self.toImGrid(dQ_n,nX,nY)

        return [X_n,dQ_n]

    def toMeshGrid(self,X,nX,nY):
        xmin = np.amin(X[:,0])
        xmax = np.amax(X[:,0])

        ymin = np.amin(X[:,1])
        ymax = np.amax(X[:,1])

        lim = (xmin,xmax,ymin,ymax)

        x = np.linspace(xmin,xmax, num=nX)
        y = np.linspace(ymin,ymax, num=nY)

        xx, yy = np.meshgrid(x, y)

        X0 = np.column_stack((xx.flatten(),yy.flatten()))

        return [X0,lim]

    def toImGrid(self,A,nX,nY):
        A_Im = np.zeros((A.shape[0],nY,nX,A.shape[2]))
        for k in range(0,A.shape[0]):
            for j in range(0,nY):
                for i in range(0,nX):
                    A_Im[k,j,i,:] = A[k,j*nX+i,:]

        return A_Im

    def interpFromMeshGrid(self,X0,X,dQ,interp='linear',fill=0.0):
        X_n = np.zeros((X.shape[0],X0.shape[0],X0.shape[1]))
        dQ_n = np.zeros((X.shape[0],X0.shape[0],dQ.shape[2]))

        for i in range(0,X.shape[0]):
            dQ_n[i] = griddata(X[i],dQ[i],X0, method=interp, fill_value = fill)
            X_n[i] = X0

        return [X_n,dQ_n]
    ##############################################################################
    #DIVIDE, EXTRACT AND SORT DATASET
    ##############################################################################
    def splitAndShuffle(self,X,dQ,alpha,testDict={},valDict={},shuffleDict={}):
        if (testDict["rand"]):
            [testX,testdQ,testAlpha,trainX,traindQ,trainAlpha] = self.randSample(testDict["rate"],X,dQ,alpha)
        else:
            [testX,testdQ,testAlpha,trainX,traindQ,trainAlpha] = self.splitDataSet(testDict["alpha"],X,dQ,alpha)

        if (shuffleDict["shuffle"]):
            [testX,testdQ,testAlpha] = self.shuffle(shuffleDict["test"],testX,testdQ,testAlpha)
            [trainX,traindQ,trainAlpha] = self.shuffle(shuffleDict["train"],trainX,traindQ,trainAlpha)

        if (valDict["validate"]):
            if (valDict["mixed"]):
                [valX,valdQ,valAlpha,trainX,traindQ,trainAlpha] = self.randSample(valDict["rate"],trainX,traindQ,trainAlpha)

            else:
                #optimise to account for shuffleDict and 
                [valX,valdQ,valAlpha,trainX,traindQ,trainAlpha] = self.splitDataSet(valDict["alpha"],trainX,traindQ,trainAlpha)

            return [trainX,traindQ,trainAlpha,valX,valdQ,valAlpha,testX,testdQ,testAlpha]

        else:
            return [trainX,traindQ,trainAlpha,testX,testdQ,testAlpha]

    def randSample(self,rate,X,dQ,alpha):
        n = X.shape[0]
        nSample = ceil(n*rate)
        IX = np.ones(n, dtype=bool)

        while(nSample > 0):
            k = randint(0,n-1)
            if (IX[k] == True):
                IX[k] = False
                nSample -= 1

        return self.extractIX(IX,X,dQ,alpha,base=False)

    def splitDataSet(self,alphaSplit,X,dQ,alpha):
        IX = np.ones(alpha.shape[0], dtype=bool)
        split = np.asarray(alphaSplit,dtype = self.dtype)
        for b in split:
            #???
            for i,a in enumerate(alpha):
                if (a==b):
                    IX[i] = False

        return self.extractIX(IX,X,dQ,alpha,base=False)

    def extractIX(self,IX,X,dQ,alpha,base = True):
        if(~base):
            ind = ~(IX)

        [X1,dQ1,alpha1] = [X[ind],dQ[ind],alpha[ind]]
        [X2,dQ2,alpha2] = [X[~(ind)],dQ[~(ind)],alpha[~(ind)]]

        return [X1,dQ1,alpha1,X2,dQ2,alpha2]

    def shuffle(self,num,X,dQ,alpha):
        n = X.shape[0]

        X_n = self.initShapeArr(X.shape,num)
        dQ_n = self.initShapeArr(dQ.shape,num)
        alpha_n = self.initShapeArr(alpha.shape,num)

        for i in range(0,num):
            arr = np.arange(X.shape[1])
            np.random.shuffle(arr)

            j = randint(0,n-1)

            X_n[i] = X[j,arr,:]
            dQ_n[i] = dQ[j,arr,:]
            alpha_n[i] = alpha[j]

        return [X_n,dQ_n,alpha_n]
    ##############################################################################
    #PROCESS SINGLE POINT DATA - FLATTEN OR REGRID
    ##############################################################################
    def flattenData(self,X,dQ,alpha):
        n = X.shape[0]
        m = X.shape[1]

        X_n = np.zeros((n*m, X.shape[2]))
        dQ_n = np.zeros((n*m, dQ.shape[2]))
        alpha_n = np.zeros((n*m,))  #???

        count = 0
        for i in range(0,n):
            for j in range(0,m):
                X_n[count,:] = X[i,j,:]
                dQ_n[count,:] = dQ[i,j,:]
                alpha_n[count,] = alpha[i]  #???
                count += 1

        return [X_n,dQ_n,alpha_n]

    ##############################################################################
    #SET AND SORT TRAINING PARAMETERS
    ##############################################################################
    def defineModelParameters(self,shapeX,shapedQ,single,grid):
        if(single):
            self.setSingleModelParameters(shapeX,shapedQ,1)
        elif(grid=="image"):
            self.setImageModelParameters(shapeX,shapedQ,1)
        else:
            self.setModelParameters(shapeX,shapedQ,1,2)

    def setModelParameters(self,shapeX,shapedQ,addData,numInputs):
        self.numInputs = numInputs

        maxPts = shapedQ[1]
        x = shapeX[2]
        q = shapedQ[2]

        self.inShape = [(addData,),(maxPts,x)]
        self.inName = ["alpha","x"]

        self.outShape = [(maxPts,q)]
        self.outName = ["dQ"]

        if ((len(self.inShape) != numInputs)or(len(self.inName) != numInputs)):
            print("Num inputs does not match length of input shape")

    def setSingleModelParameters(self,shapeX,shapedQ,addData):
        #NEED TO CHECK IF CORRECT, ROBUST FOR MORE ACCURATE
        lX = list(shapeX)
        lQ = list(shapedQ)

        lX.append(1)
        lQ.append(1)

        shapeX = tuple(lX)
        shapedQ = tuple(lQ)

        self.setModelParameters(shapeX,shapedQ,addData,2)

    def setImageModelParameters(self,shapeX,shapedQ,addData):
        #NEED TO CHECK IF CORRECT, ROBUST FOR MORE ACCURATE
        self.numInputs = 2

        x = shapeX[3]
        nY = shapedQ[1]
        nX = shapedQ[2]
        q = shapedQ[3]

        self.inShape = [(addData,),(nY,nX,x)]
        self.inName = ["alpha","x"]

        self.outShape = [(nY,nX,q)]
        self.outName = ["dQ"]

        if ((len(self.inShape) != self.numInputs)or(len(self.inName) != self.numInputs)):
            print("Num inputs does not match length of input shape")

    def initShapeArr(self,shape,num):
        temp = list(shape)
        temp[0] = num
        shape = tuple(temp)

        return np.zeros((shape))

    def getModelParameters(self):
        return [self.numInputs,self.inShape,self.inName,self.outShape,self.outName]
    ##############################################################################
    #OTHER
    ##############################################################################
    def calculateDelta(self,X,X0,Q,Q0,interp='linear',fill=0.0):
        #Qn = griddata(X,Q,X0, method=interp, fill_value = fill)
        Qn = griddata(X,Q,X0, method=interp)#, fill_value = nan)
        deltaQ = np.subtract(Qn,Q0)

        deltaQ[np.isnan(deltaQ)] = 0.0

        return deltaQ
    
    def regrid(self,alpha0=1.0):
        ind = self.findAlpha(alpha0)

        Qn = []
        Xn = []
        for i in range(0,self.numHills):
            Qn.append(griddata(self.X[i],self.Q[i],self.X[ind], method='linear', fill_value = 0.0))
            Xn.append(self.X[ind])

        Qn = np.asarray(Qn, dtype=self.dtype)
        Xn = np.asarray(Xn, dtype=self.dtype)

        return [Xn,Qn]

    def getVol(self,alpha0=1.0):
        ind = self.findAlpha(alpha0)
 
        vol = self.hills[ind].getVolume()

        vol = np.asarray(vol, dtype=self.dtype)

        #NEED TO DEFINE MORE ROBUSTLY
        #vol /= np.amax(np.abs(vol))

        return vol
    ##############################################################################
    #DATASET PROCESS TREE
    ##############################################################################
    def getTrainingDict(self):
        testDict = {"rand": False, "rate": 0.15, "alpha": [0.8]}
        valDict = {"validate": False, "mixed": False, "rate": 0.2, "alpha": [1.2]}
        shuffleDict = {"shuffle": False, "train": 200, "val": 40, "test": 20}
        modelDict = {"difference":"fixed","alpha0":1.0
                     ,"remHill":False,"subset":False
                     ,"xMin":float('-inf'),"xMax":float('inf')
                     ,"yMin":float('-inf'),"yMax":float('inf')
                     ,"normalX":"none","normaldQ":"abs"
                     ,"grid":"base","nX":200,"nY":160
                     ,"random":False,"single":False}

        return [modelDict, testDict, valDict, shuffleDict]
    
    def getData(self,modelDict={},testDict={},valDict={},shuffleDict={}):
        [X,dQ] = self.generateDelta(modelDict["difference"],modelDict["alpha0"])

        [X,dQ] = self.removeUnderHill(X,dQ,modelDict["remHill"])

        #[X,dQ] = self.addBoundaryPoints(X,dQ,modelDict["boundary"],modelDict["bcPoints"])
        
        alpha = self.alpha

        [X,dQ] = self.getSubset(X,dQ,modelDict["subset"],modelDict["xMin"],modelDict["xMax"],modelDict["yMin"],modelDict["yMax"])

        X = self.normalise(X,modelDict["normalX"])
        dQ = self.normalise(dQ,modelDict["normaldQ"])

        #RANDOM SQUARE
        #ACCOUNT FOR ARBITARY RANDOM SQUARE
        [X,dQ] = self.changeGrid(X,dQ,modelDict["grid"],modelDict["nX"],modelDict["nY"],modelDict["alpha0"])
        #CONVOLUTE OR RANDOM SQURARE

        #SPLIT AND SHUFFLE - need to robust for certain combinations #SPLIT SEPERATE FROM SHUFFLE??
        if (valDict["validate"]):
            [trainX,traindQ,trainAlpha,valX,valdQ,valAlpha,testX,testdQ,testAlpha] = self.splitAndShuffle(X=X,dQ=dQ,alpha=alpha,testDict=testDict,valDict=valDict,shuffleDict=shuffleDict)
        else:
            [trainX,traindQ,trainAlpha,testX,testdQ,testAlpha] = self.splitAndShuffle(X=X,dQ=dQ,alpha=alpha,testDict=testDict,valDict=valDict,shuffleDict=shuffleDict)
        
        #######################################################################################
        #REFACTOR THIS REGION
        if(modelDict["single"]):
            [testX,testdQ,testAlpha] = self.flattenData(testX,testdQ,testAlpha)
            [trainX,traindQ,trainAlpha] = self.flattenData(trainX,traindQ,trainAlpha)
            if (valDict["validate"]):
                [valX,valdQ,valAlpha] = self.flattenData(valX,valdQ,valAlpha)

        self.defineModelParameters(trainX.shape,traindQ.shape,modelDict["single"],modelDict["grid"])

        #REFACTOR THIS REGION
        if(modelDict["single"]):
            testX = self.sepVar(testX)
            testdQ = self.sepVar(testdQ)

            trainX = self.sepVar(trainX)
            traindQ = self.sepVar(traindQ)

            if (valDict["validate"]):
                valX = self.sepVar(valX)
                valdQ = self.sepVar(valdQ)

        #######################################################################################

        if (valDict["validate"]):
            return [trainX,traindQ,trainAlpha,valX,valdQ,valAlpha,testX,testdQ,testAlpha]
        else:
            return [trainX,traindQ,trainAlpha,testX,testdQ,testAlpha]

    def sepVar(self,X):
        X_n = []

        for i in range(0,X.shape[1]):
            X_n.append(X[:,i])

        return X_n
    ##############################################################################
    #CALL FOR CORRECT DATASETS
    ##############################################################################
    def getFullHillDataTest(self, testDict={},valDict={},shuffleDict={}):
        #turn to numpy and generate deltas at init or otherwise
  
        [X,dQ] = self.generateBackwardDifference()
        [ind1,ind2] = self.findBDIndex()
        alpha = np.column_stack((self.alpha[ind1,],self.alpha[ind2,]))

        self.setModelParameters(X.shape,dQ.shape,2,2)
        return self.splitAndShuffle(X=X,dQ=dQ,alpha=alpha,testDict=testDict,valDict=valDict,shuffleDict=shuffleDict)

        #Generalise for 3D space
    def getRandomSquareData(self, alpha0=1.0,dx=0.5,dy=0.5,nx=10,ny=10,num=20,testDict={},valDict={},shuffleDict={}):
        #turn to numpy and generate deltas at init or otherwise
  
        [X,dQ] = self.generateFixedDifference(alpha0)
        alpha = self.alpha

        [X,dQ,alpha] = self.generateRandomSquares(X,dQ,alpha,dx=dx,dy=dy,nx=nx,ny=ny,num=num)

        self.setModelParameters(X.shape,dQ.shape,1,2)

        return self.splitAndShuffle(X=X,dQ=dQ,alpha=alpha,testDict=testDict,valDict=valDict,shuffleDict=shuffleDict)

    ##############################################################################
    #DIVIDE, EXTRACT AND SORT DATASET
    ##############################################################################
    def generateRandomSquares(self,X,dQ,alpha,dx,dy,nx,ny,num):
        X_n = []
        dQ_n = []
        alpha_n = []
        xMin = np.amin(X[:,:,0])
        xMax = np.amax(X[:,:,0])
        yMin = np.amin(X[:,:,1])
        yMax = np.amax(X[:,:,1])

        #enumerate
        for i in range(0,X.shape[0]):
            xMin = np.amin(X[i,:,0])
            xMax = np.amax(X[i,:,0])
            yMin = np.amin(X[i,:,1])
            yMax = np.amax(X[i,:,1])

            for n in range(0,num):
                x = uniform(xMin, xMax-dx)
                y = uniform(yMin, yMax-dy)

                ind = ((X[i,:,0] >= x) & (X[i,:,0] <= (x+dx)) & (X[i,:,1] >= y) & (X[i,:,1] <= (y+dy)))

                X_temp = X[i,ind,:]
                dQ_temp = dQ[i,ind,:]

                if(X_temp.shape[0] > 0):
                    #print(X_temp.shape[0])
                    minX = np.amin(X_temp[:,0])
                    maxX = np.amax(X_temp[:,0])
                    minY = np.amin(X_temp[:,1])
                    maxY = np.amax(X_temp[:,1])

                    #xArr = np.linspace(minX, maxX, nx)
                    #yArr = np.linspace(minY, maxY, ny)

                    xArr = np.linspace(x, x+dx, nx)
                    yArr = np.linspace(y, y+dy, ny)

                    xGrid, yGrid = np.meshgrid(xArr, yArr)
                    X0 = np.column_stack((xGrid.flatten(),yGrid.flatten()))

                    X_n.append(X0)

                    dQ_n.append(griddata(X_temp,dQ_temp,X0, method='linear', fill_value = 0.0))

                    alpha_n.append(alpha[i])

        X_n = np.asarray(X_n, dtype = self.dtype)
        dQ_n = np.asarray(dQ_n, dtype = self.dtype)

        alpha_n = np.asarray(alpha_n, dtype = self.dtype)
        alpha_n = alpha_n[:,]

        return [X_n,dQ_n,alpha_n]
    ##############################################################################
    #PROCESS SINGLE POINT DATA - FLATTEN OR REGRID
    ##############################################################################
    def reshapeData(self,X,dQ,alpha):
        n = X.shape[0]

        X_n = []
        dQ_n = []
        alpha_n = []

        for i in range(0,n):
            a = alpha[i]
            found_alpha = False
            IX = 0
            for j in range(0,len(alpha_n)):
                if (a == alpha_n[j]):
                    found_alpha = True
                    IX = j
                    break

            if (found_alpha == True):
                X_n[IX].append(X[i])
                dQ_n[IX].append(dQ[i])
            else:
                X_temp = []
                X_temp.append(X[i])
                X_n.append(X_temp)

                dQ_temp = []
                dQ_temp.append(dQ[i])
                dQ_n.append(dQ_temp)

                alpha_n.append(a)

        X_n = np.asarray(X_n, dtype = self.dtype)
        dQ_n = np.asarray(dQ_n, dtype = self.dtype)

        alpha_n = np.asarray(alpha_n, dtype = self.dtype)
        alpha_n = alpha_n[:,]

        return [X_n,dQ_n,alpha_n]

    def processSingleData(self,X,truthdQ,predictdQ,alpha):
        predictdQ=np.squeeze(predictdQ,axis=2)
        [X_n,truthdQ,alpha_n] = self.reshapeData(X,truthdQ,alpha)
        [X,predictdQ,alpha] = self.reshapeData(X,predictdQ,alpha)

        return [X,truthdQ,predictdQ,alpha]
    ##############################################################################
    #GET FLOWFIELD LINEARITY WRT PARAMETER
    ##############################################################################
    def findR(self,alpha,y):
        gradient, intercept, r, p_value, std_err = linregress(alpha,y)

        return r

    def createRField(self, alpha0=1.0):
        [X,dQ] = self.generateFixedDifference(alpha0)
        alpha = self.alpha

        n = X.shape[1]
        m = dQ.shape[2]
        l = X.shape[0]

        R = np.zeros((n,m))
        for i in range(0,n):
            for j in range(0,m):
                y = []
                for k in range(0,l):
                    y.append(dQ[k,i,j])
                R[i,j] = self.findR(alpha,y)
        return [X[0],R]
    ##############################################################################
    #GET FLOW FIELD GRADIENTS
    ##############################################################################
    def addGrad(self,dX):
        for i,hill in enumerate(self.hills):
            hill.addGrad(dX)
            self.hills[i] = hill

        self.toNumpy()
    ##############################################################################
    #CONVOLUTIONAL DATA CALCULATIONS
    ##############################################################################
    def convData(self,X,dQ,nX,nY,alpha0=1.0,interp='linear',fill=0.0):
        [X0] = self.getHillFromAlpha(alpha0,True,False)

        [X0,lim] = self.toMeshGrid(X0,nX,nY)
        dQ = self.toImage(X0,X,dQ,nX,nY,interp,fill)

        return [X0,dQ,lim]

    def toImage(self,X,dQ,nX,nY):
        dQ_n = np.zeros((dQ.shape[0],nY,nX,dQ.shape[2]))
        for i in range(0,dQ.shape[0]):
            dQ_n[i] = self.interpRectGrid(X0,X[i],dQ[i],nX,nY,interp,fill)

        return dQ_n

    def interpRectGrid(self,X0,X,dQ,nX,nY,interp='linear',fill=0.0):
        dQ0 = interpolate.griddata(X,dQ,X0, method=interp, fill_value = fill)
        dQ0 = self.toRectGridQ(dQ0,nX,nY)

        return dQ0

    def toRectGridQ(self,dQ,nX,nY):
        dQRect = np.zeros((nY,nX,dQ.shape[1]))
        for j in range(0,nY):
            for i in range(0,nX):
                dQRect[j,i,:] = dQ[j*nX+i,:]

        return dQRect