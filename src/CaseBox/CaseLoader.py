from PDEBox.PDELoader import PDELoader

class CaseLoader:
    def __init__(
        self,
        case):

        self.case = case

        self.domain = 0
        self.bc_list = []
        self.bc_name = []
        self.data = 0

        self.pde_store = 0
        self.pde_prop = []

        self.initGeometry()

        ## Next Steps: generating BC; storing and setting up PINN (architecture,weights,iterations etc)

    def initGeometry(self):
        return

    def setPDE(self, pde_type, pde_id):
        
        self.pde_store = PDELoader(pde_type, pde_id)
        self.pde_store.setPDE(self.pde_prop)

        return

    def genBC(self):
        self.bc_list = []
        return

##########################################################
    def getPDE(self):
        return self.pde_store.getPDE()

    def getNetShape(self):
        return self.pde_store.getNetShape()

    def getPDEName(self):
        return self.pde_store.getPDEName()
###########################################################
    def loadData(self):
        data = []
        return
###########################################################