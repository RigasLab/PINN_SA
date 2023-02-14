import numpy as np

# This class stores data (location, variables, etc.) for a single point
# Inputs:
#       data        -       Data at point in string format, includes X and any Q
#       headers     -       Headers(variable names) from the raw file - must be same length as data
class Point:
    def __init__(
        self,
        data,
        header,
        dtype = np.float32
        ):
        self.dtype = dtype

        # Dictionary storing values of x,y,z initialised as 0.0
        self.X = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0
        }

        # Dictionary to store flow variable such as U,p and turbulence parameters
        self.Q = {}
        self.vol = 0.0
        # Take each value in data(w/ header) and store in respective dict
        # If the header is a coordinate store in X else store in Q
        for h,val in zip(header,data):
            if h in self.X :
                self.X.update({h:float(val)})
            elif (h == "Quality"):
                self.vol = float(val)
            else:
                self.Q.update({h:float(val)})

    # Returns X
    def getX(self):
        return self.X

    # Returns X keys
    def getXKeys(self):
        return self.X.keys()

    # Returns Q
    def getQ(self):
        return self.Q

    # Returns cell volume
    def getVol(self):
        return self.vol
    
    # Returns Q keys
    def getQKeys(self):
        return self.Q.keys()

    # NEW - NEEDS TESTING
    def addToQ(self,header,data):
        for h,val in zip(header,data):
            self.Q.update({h:float(val)})


    # This function checks if the point lies on a 2D plane with axis and station
    # Inputs:
    #       axis    -       string ("x","y","z") with axis normal to 2D plane
    #       pos     -       station of 2D plane(on axis)
    # Outputs:
    #       boolean
    def is2D(self, axis = "z", pos = 0.0):
        if (self.X[axis] == pos):
            return True
        else:
            return False

    # This function removes from the variables specific to an axis to reduce dimension
    # Take away the X dict entry as well as corresponding Q velocity component
    # Inputs:
    #       axis    -       string ("x","y","z") with axis normal to 2D plane
    #
    # Currently modifies point - return modified copy instead?
    # What happens if velocity component not present in q?
    def extract2D(self, axis = "z"):
        #Map representing axis and velocity component - check speed if in init vs global  not
        axisMap = {
            "x": "u",
            "y": "v",
            "z": "w"
        }
        # Remove relevant entries from X and Q
        self.X.pop(axis)

        if (axisMap[axis] in self.Q.keys()):
            self.Q.pop(axisMap[axis])
    
    # This function removes any non velocity/pressure components from Q
    def removeTurbulence(self):
        # Keys which represent velocity and pressure
        flowSet = {"u", "v", "w", "p"}
        # Collect keys from Q and determine the difference(non vel/pressure keys)
        keys = set(self.Q.keys())
        keys.difference_update(flowSet)

        # remove the relevant keys
        for key in keys:
            self.Q.pop(key)

    # Returns X as a numpy in a specific order(x,y,z)
    def xToNumpy(self):
        return np.asarray(list(map(self.X.get, self.xMap())), dtype=self.dtype)

    #[mydict.get(key) for key in keys if key in mydict]
    #[myDictionary.get(key) for key in keys]

    # Returns Q as a numpy in specific order (u,v,w,p, anything else)
    def qToNumpy(self):
        return np.asarray(list(map(self.Q.get, self.qMap())), dtype=self.dtype)

    # This function returns a map of keys representing the specific output order of x
    # Uses typical x,y,z order. Removes any keys not in data 
    def xMap(self):
        full_map = ["x", "y", "z"]
        map = []

        for m in full_map:
            if m in self.X:
                map.append(m)

        return map

    # This function returns a map of keys representing the specific output order of Q
    # Uses typical u,v,w,p et al. order. Removes any keys not in data 
    def qMap(self):
        full_map = ["u", "v", "w", "p"]
        map = []

        for m in full_map:
            if m in self.Q:
                map.append(m)

        for q in self.Q.keys():
            if q not in map:
                map.append(q)

        return map

    # This function normalises velocity and pressure
    def normalise(self,velNorm,presNorm):
        self.normaliseVel(velNorm)
        self.normalisePres(presNorm)

    # This function normalises velocity
    def normaliseVel(self,velNorm):
        keys = ["u","v","w"]
        for key in keys:
            self.Q[key] /= velNorm

    def normalisePres(self,presNorm):  
        self.Q["p"] /= presNorm

    def normaliseQ(self,key,norm):
        self.Q[key]/= norm

    def normaliseX(self,key,norm):
        self.X[key]/= norm