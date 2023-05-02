from GeometryBox.PeriodicHill.PHXDEGeometryBase import PHXDEGeometryBase

class PHLoader(Casebox.CaseLoader):
    def __init__(
        self,
        case,
        Re,
        rho = 1.0,
        H = 1.0,
        U = 1.0,
        fixedfp = True,
        fp = -0.0110,
        minS = 1e-10
        ):

        super(PHLoader, self).__init__(
            case
            )

        self.Re = Re
        self.rho = rho
        self.U = U
        self.H = H
        self.nu = U*H/Re

        self.fp = fp
        self.initfp(fixedfp)

        self.minS = minS

        self.pde_prop = [self.rho, self.nu, self.fp, self.minS]

        self.X_test = []
        self.Q_test = []
        self.tau_test = []

        self.data_path = ""


        #self.domain = 0
        #self.bc_list = []
        #self.bc_name = []
        #self.data = 0
        #self.pde_store = 0

###########################################
    def initGeometry(self):
        self.domain = PHXDEGeometryBase()

        return

    def initfp(self,fix):
        if (fix == False):
            #self.fp = dde.Variable(self.fp, dtype=dde.config.default_float())
            self.fp = self.fp

        return

    def loadData(self, data_path):
        self.data_path = data_path

        removeUpper = True

        refineUpper = {"refine":False,
                       "res":10,
                       "x0":0.0}
        #############################################################
        [_,_,_,self.X_test,self.Q_test,self.tau_test] = loadQ0(data_path,10,5,refineUpper,removeUpper)

        return




