from PDEBox.RANS import RANS_B_RSTE,RANS_B_HD,RANS_SA_HD

class PDELoader:
    def __init__(
        self,
        pde_type,
        pde_id
        ):

        self.pde_type = pde_type
        self.pde_id = pde_id

        self.pde = 0
        self.netShape = [1,1]
        self.pde_name = []


    def setPDE(self,prop):
        if (self.pde_type == "RANS"):
            self.setRANS(prop)

        else:
            print("PDE Type not found")
            quit()

        return


    def setRANS(self,prop):
        #prop = [rho,nu,minS,fp]
        RANS_form = self.pde_id[0]
        doSA = self.pde_id = [1]

        if (doSA):
            if (RANS_form == "RST-E"):
                print("RST-E with SA not configured")
                self.netShape = [2,7]
                quit()   
            elif (RANS_form == "HD"):          
                self.pde = RANS_SA_HD(prop[0],prop[1],prop[3],prop[2])
                self.netShape = [2,6]
        else:
            if (RANS_form == "RST-E"):
                self.pde = RANS_B_RSTE(prop[0],prop[1],prop[2])
                self.netShape = [2,6]
            elif (RANS_form == "HD"):          
                self.pde = RANS_B_HD(prop[0],prop[1],prop[2])
                self.netShape = [2,5]

        self.setRANSName(self,RANS_form,doSA)

        return

    def setRANSName(self,RANS_form,doSA):
        self.pde_name.append('mass')
        self.pde_name.append('x-mom')
        self.pde_name.append('y-mom')

        if (RANS_form == "HD"):
            self.pde_name.append('div(fs)')

        if (doSA):
            self.pde_name.append('SA')
            self.pde_name.append('minF')

        return
    
    def getPDE(self):
        return self.pde
       
    def getNetShape(self):
        return self.netShape
    
    def getPDEName(self):
        return self.pde_name

