from GeometryBox.PeriodicHill.PHXDEGeometryBase import PHXDEGeometryBase

class PHLoader(Casebox.CaseLoader):
    def __init__(
        self,
        case,
        Re,
        rho = 1.0,
        H = 1.0,
        U = 1.0
        ):

        self.Re = Re
        self.rho = rho
        self.U = U
        self.H = H
        self.nu = U*H/Re

        self.initGeometry()


        self.domain = []
        self.bc_list = []
        self.bc_name = []
        self.pde = []
        self.pde_name = []
        self.data = []
        self.netShape = []


    def initGeometry(self):
        self.domain = PHXDEGeometryBase()

        return

    def setPDE(self, type, SA):
        if (doSA):
            if (type == "RST-E"):
                print("RST-E with SA not configured")
                self.netShape = [2,7]
                quit()   
            elif (type == "HD"):          
                self.pde = RANS_SA_HD(self.rho,self.nu,self.minS,self.fp)
                self.netShape = [2,6]
        else:
            if (type == "RST-E"):
                pde = RANS_B_RSTE(rho,nu,fp)
                self.netShape = [2,6]
            elif (type == "HD"):          
                pde = RANS_B_HD(rho,nu,fp)
                self.netShape = [2,5]

    def getNetShape(self):
        return self.netShape

    def getPDE(self):
        return self.pde