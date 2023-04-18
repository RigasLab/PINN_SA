
class CaseLoader:
    def __init__(
        self,
        case):

        self.geometry = []
        self.bc_list = []
        self.bc_name = []
        self.pde = []
        self.pde_name = []
        self.data = []
        self.netShape = []


    def initGeometry(self):
        return

    def getNetShape(self):
        return self.netShape

    def getPDE(self):
        return self.pde