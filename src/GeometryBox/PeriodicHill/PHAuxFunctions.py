import numpy as np

#####################################################################################################################
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

def genHArr(X):
    h = np.zeros(X.shape)

    for i,x in enumerate(X):
        h[i] = genH(x)

    return h

def getBSLHill(N=1000):
    xh = np.linspace(0,9,N)
    yh = genHArr(xh)

    return [xh,yh]

def getHillSegment(xa,xb,N=1000):
    xh = np.linspace(xa,xb,N)
    yh = np.zeros(N)

    for i,x in enumerate(xh):
        yh[i] = genH(x)
    return [xh,yh]

def getD(X):
    du = 3.036 - X[:,1:2]
    [xh,yh] = getHillSegment(0, 9, 10000)

    d = np.zeros(du.shape)

    for i in range(0,d.shape[0]):

        if(X[i,1:2] >= 3.036/2 + 0.5):
            d[i,:] = du[i,:]
        else:
            dAll = (X[i,0:1] + 4.5 - xh)**2 + (X[i,1:2]-yh)**2
            dl = np.sqrt(np.amin(dAll))

            d[i,:] = np.amin([du[i,:],dl])

    return d

