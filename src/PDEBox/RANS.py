import deepxde as dde
import numpy as np

from deepxde.backend import tf
###################################################################################
###################################################################################
#RANS_B_RST-E
def RANS_B_RSTE(rho,nu,fp):
    """
    Return PDE for RANS with RST-E formulation.

    :param rho: Fluid Density.
    :type rho: float
    :param nu: Fluid Kinematic Viscosity.
    :type nu: float
    :param fp: Streamwise Body Forcing.
    :type fp: float or tf.Variable
    :return: PDE operator to calculate residual of continuity, x-momentum and y-momentum
    :rtype: func

    """
    def pde(X,Q,D):
        """
        Returns PDE loss for RST-E formulation.

        :param X: Coordinates to evaluate residual error.
        :type X: numpy or tf.Tensor
        :param Q: Corresponding mean flow variables.
        :type Q: numpy or tf.Tensor
        :param D: Auxillar variable corresponding to wall distance.
        :type D: None or numpy or tf.Tensor
        :return: PDE loss corresponding with X for continuity, x-momentum and y-momentum
        :rtype: list

        """
        U       = Q[:,0:1]
        V       = Q[:,1:2]
        P       = Q[:,2:3]

        uu      = Q[:,3:4]
        uv      = Q[:,4:5]
        vv      = Q[:,5:6]
        ####################################################
        Ux = dde.grad.jacobian(Q, X, i=0, j=0)
        Uy = dde.grad.jacobian(Q, X, i=0, j=1)

        Vx = dde.grad.jacobian(Q, X, i=1, j=0)
        Vy = dde.grad.jacobian(Q, X, i=1, j=1)

        Px = dde.grad.jacobian(Q, X, i=2, j=0)
        Py = dde.grad.jacobian(Q, X, i=2, j=1)

        Uxx = dde.grad.hessian(Q, X, component=0, i=0, j=0)
        Uyy = dde.grad.hessian(Q, X, component=0, i=1, j=1)

        Vxx = dde.grad.hessian(Q, X, component=1, i=0, j=0)
        Vyy = dde.grad.hessian(Q, X, component=1, i=1, j=1)

        uux = dde.grad.jacobian(Q, X, i=3, j=0)
        uvy = dde.grad.jacobian(Q, X, i=4, j=1)

        uvx = dde.grad.jacobian(Q, X, i=4, j=0)
        vvy = dde.grad.jacobian(Q, X, i=5, j=1)
        #####################################################
        #####################################################
        cont  = Ux + Vy
        xmom  = U*Ux + V*Uy + (1/rho)*Px - nu*(Uxx + Uyy) + uux + uvy + fp
        ymom  = U*Vx + V*Vy + (1/rho)*Py - nu*(Vxx + Vyy) + uvx + vvy

        return [cont, xmom, ymom]
    return pde
###################################################################################
###################################################################################
#RANS_B_HD
def RANS_B_HD(rho,nu,fp):
    """
    Return PDE for RANS with HD formulation and no turbulence model.

    :param rho: Fluid Density.
    :type rho: float
    :param nu: Fluid Kinematic Viscosity.
    :type nu: float
    :param fp: Streamwise Body Forcing.
    :type fp: float or tf.Variable
    :return: PDE operator to calculate residual of continuity, x-momentum, y-momentum and divergence of solenoidal forcing.
    :rtype: func

    """
    def pde(X,Q,D):
        """
        Returns PDE loss for HD formulation (without Spalart-Allmaras)

        :param X: Coordinates to evaluate residual error.
        :type X: numpy or tf.Tensor
        :param Q: Corresponding mean flow variables.
        :type Q: numpy or tf.Tensor
        :param D: Auxillar variable corresponding to wall distance.
        :type D: None or numpy or tf.Tensor
        :return: PDE loss corresponding with X for continuity, x-momentumm, y-momentum and divergence of solenoidal forcing.
        :rtype: list

        """
        U       = Q[:,0:1]
        V       = Q[:,1:2]
        Pphi    = Q[:,2:3]
        
        fsu     = Q[:,3:4]
        fsv     = Q[:,4:5]
        ####################################################
        Ux = dde.grad.jacobian(Q, X, i=0, j=0)
        Uy = dde.grad.jacobian(Q, X, i=0, j=1)

        Vx = dde.grad.jacobian(Q, X, i=1, j=0)
        Vy = dde.grad.jacobian(Q, X, i=1, j=1)

        Pphix = dde.grad.jacobian(Q, X, i=2, j=0)
        Pphiy = dde.grad.jacobian(Q, X, i=2, j=1)

        fsux = dde.grad.jacobian(Q, X, i=3, j=0)
        fsvy = dde.grad.jacobian(Q, X, i=4, j=1)

        Uxx = dde.grad.hessian(Q, X, component=0, i=0, j=0)
        Uyy = dde.grad.hessian(Q, X, component=0, i=1, j=1)

        Vxx = dde.grad.hessian(Q, X, component=1, i=0, j=0)
        Vyy = dde.grad.hessian(Q, X, component=1, i=1, j=1)
        #####################################################
        #####################################################
        cont  = Ux + Vy
        xmom  = U*Ux + V*Uy + (1/rho)*Pphix - nu*(Uxx + Uyy) - fsu + fp
        ymom  = U*Vx + V*Vy + (1/rho)*Pphiy - nu*(Vxx + Vyy) - fsv
        divfs = fsux + fsvy

        return [cont, xmom, ymom, divfs]
    return pde
###################################################################################
###################################################################################
##RANS_SA_HD
def RANS_SA_HD(rho,nu,minS,fp):
    """
    Return PDE for RANS with HD formulation and Spalart-Allmaras turbulence model.

    :param rho: Fluid Density.
    :type rho: float
    :param nu: Fluid Kinematic Viscosity.
    :type nu: float
    :param minS: Minimum value of mean strain rate tensor to prevent division by 0.
    :type minS: float
    :param fp: Streamwise Body Forcing.
    :type fp: float or tf.Variable
    :return: PDE operator to calculate residual of continuity, x-momentum, y-momentum, divergence of solenoidal forcing, Spalart-Allmaras transport equation and L2 norm of solenoidal forcing.
    :rtype: func

    """
    cv1 = 7.1
    cv2 = 0.7
    cv3 = 0.9

    kep = 0.41 

    cb1 = 0.1355
    cb2 = 0.622
    sig = 2/3

    cw1 = (cb1/(kep*kep)) + (1+cb2)/sig
    cw2 = 0.3
    cw3 = 2.0
    rlim = 10.0

    M = 1e-5

    def pde(X,Q,D):
        """
        Returns PDE loss for HD formulation (with Spalart-Allmaras)

        :param X: Coordinates to evaluate residual error.
        :type X: numpy or tf.Tensor
        :param Q: Corresponding mean flow variables.
        :type Q: numpy or tf.Tensor
        :param D: Auxillar variable corresponding to wall distance.
        :type D: None or numpy or tf.Tensor
        :return: PDE loss corresponding with X for continuity, x-momentumm, y-momentum, divergence of solenoidal forcing, Spalart-Allmaras transport equation and L2 norm of solenoidal forcing.
        :rtype: list

        """
        U       = Q[:,0:1]
        V       = Q[:,1:2]
        Pphi    = Q[:,2:3]
        
        nutilde = Q[:,3:4]   
        fsu     = Q[:,4:5]
        fsv     = Q[:,5:6]
        ####################################################
        Ux = dde.grad.jacobian(Q, X, i=0, j=0)
        Uy = dde.grad.jacobian(Q, X, i=0, j=1)

        Vx = dde.grad.jacobian(Q, X, i=1, j=0)
        Vy = dde.grad.jacobian(Q, X, i=1, j=1)

        Pphix = dde.grad.jacobian(Q, X, i=2, j=0)
        Pphiy = dde.grad.jacobian(Q, X, i=2, j=1)

        nutildex = dde.grad.jacobian(Q,X, i=3, j=0)
        nutildey = dde.grad.jacobian(Q,X, i=3, j=1)

        fsux = dde.grad.jacobian(Q, X, i=4, j=0)
        fsvy = dde.grad.jacobian(Q, X, i=5, j=1)

        Uxx = dde.grad.hessian(Q, X, component=0, i=0, j=0)
        Uyy = dde.grad.hessian(Q, X, component=0, i=1, j=1)

        Vxx = dde.grad.hessian(Q, X, component=1, i=0, j=0)

        Vyy = dde.grad.hessian(Q, X, component=1, i=1, j=1)

        nutildexx = dde.grad.hessian(Q, X, component=3, i=0, j=0)
        nutildeyy = dde.grad.hessian(Q, X, component=3, i=1, j=1)
        #####################################################
        #####################################################
        dn = D[:,0:1]
        d = tf.reshape(dn,tf.shape(nutilde))
        dsqr = d*d
        ############################
        chi = nutilde/nu

        fv1 = (chi**3)/((chi**3) + cv1**3)
        ###################################################
        nut = fv1*tf.math.maximum(tf.zeros(shape=tf.shape(nutilde),dtype = tf.dtypes.as_dtype(dde.config.default_float())), nutilde)

        nutx = dde.grad.jacobian(nut,X, i=0, j=0)
        nuty = dde.grad.jacobian(nut,X, i=0, j=1)
        ###################################################     
        eta = nu*tf.where(nutilde < 0.0,(1.0 + chi + 0.5*chi**2), (1.0 + chi))

        etax = dde.grad.jacobian(eta,X, i=0, j=0)
        etay = dde.grad.jacobian(eta,X, i=0, j=1)
        ###################################################
        vorZ = Vx - Uy
        Sdash = tf.math.sqrt(vorZ*vorZ)
        S_dsqr = dsqr*(tf.math.sqrt(Sdash*Sdash + M*M) - M)

        fv2 = 1 - chi/(1 + chi*fv1)
        ####################################################
        Sbar_dsqr = nutilde*fv2/(kep*kep)
        Stilde_dsqr = tf.math.maximum(minS*tf.ones(shape=tf.shape(S_dsqr),dtype = tf.dtypes.as_dtype(dde.config.default_float())), (S_dsqr + Sbar_dsqr))
        ###################################################
        rn = nutilde/(Stilde_dsqr*kep*kep)
        r = tf_Stilde(rn)
        ###################################################
        g = r + cw2*(r**6 - r)

        fw = g*((1 + cw3**6)/((g**6) + cw3**6))**(1/6)
        ###################################################
        gn = 1 - (1000*chi**2)/(1+chi**2)
        ##################################################
        ##################################################
        sdiff = (1/sig)*(eta*(nutildexx + nutildeyy) + etax*nutildex + etay*nutildey)
        
        sp_dsqr = tf.where(nutilde < 0.0, (cb1*S_dsqr*nutilde*gn), (cb1*Stilde_dsqr*nutilde))
        
        sd_dsqr = tf.where(nutilde < 0.0, (cw1*nutilde*nutilde), (-cw1*fw*nutilde*nutilde))
        
        sc = (cb2/sig)*(nutildex*nutildex + nutildey*nutildey)
        #####################################################
        #####################################################
        cont  = Ux + Vy
        xmom  = U*Ux + V*Uy + (1/rho)*Pphix - (nu+nut)*(Uxx + Uyy) - 2*nutx*Ux - nuty*(Uy + Vx) - fsu + fp
        ymom  = U*Vx + V*Vy + (1/rho)*Pphiy - (nu+nut)*(Vxx + Vyy) - 2*nuty*Vy - nutx*(Uy + Vx) - fsv
        divfs = fsux + fsvy
        SA   = dsqr*(U*nutildex + V*nutildey - sdiff - sc) - sp_dsqr - sd_dsqr
        minF  = tf.math.sqrt(0.5*(fsu*fsu + fsv*fsv))

        return [cont, xmom, ymom, divfs, SA, minF]
    return pde
########################################################################
########################################################################
def getSATerms(rho,nu,dmin=1e-5):
    """
    Return function to evaluate teerms in SA model.

    :param rho: Fluid Density.
    :type rho: float
    :param nu: Fluid Kinematic Viscosity.
    :type nu: float
    :param dmin: Minimum value of wall distance for blend function.
    :type dmin: float
    :return: Operator to calculate production, destruction, diffusion and cross-diffusion terms.
    :rtype: func

    """
    cv1 = 7.1
    cv2 = 0.7
    cv3 = 0.9

    kep = 0.41 

    cb1 = 0.1355
    cb2 = 0.622
    sig = 2/3

    cw1 = (cb1/(kep*kep)) + (1+cb2)/sig
    cw2 = 0.3
    cw3 = 2.0
    rlim = 10.0

    M = 1e-5

    ep = dmin
    a = 1/dmin
    m = 0.25
    def pde(X,Q,D):
        U       = Q[:,0:1]
        V       = Q[:,1:2]
        nutilde = Q[:,3:4]
        ####################################################
        Ux = dde.grad.jacobian(Q, X, i=0, j=0)
        Uy = dde.grad.jacobian(Q, X, i=0, j=1)

        Vx = dde.grad.jacobian(Q, X, i=1, j=0)
        Vy = dde.grad.jacobian(Q, X, i=1, j=1)

        nutildex = dde.grad.jacobian(Q,X, i=3, j=0)
        nutildey = dde.grad.jacobian(Q,X, i=3, j=1)

        nutildexx = dde.grad.hessian(Q, X, component=3, i=0, j=0)
        nutildeyy = dde.grad.hessian(Q, X, component=3, i=1, j=1)
        #####################################################
        #####################################################
        dn = D[:,0:1]
        db = dn*tf.math.tanh(a*dn) + (1.0 - tf.math.tanh(m*a*dn))*ep
        d = tf.reshape(db,tf.shape(nutilde))

        inv_dsqr = tf.math.reciprocal_no_nan(d*d,name = "invertdsqr")
        ############################
        chi = nutilde/nu

        fv1 = (chi**3)/((chi**3) + cv1**3)
        ###################################################
        nut = fv1*tf.math.maximum(tf.zeros(shape=tf.shape(nutilde),dtype = tf.dtypes.as_dtype(dde.config.default_float())), nutilde)

        nutx = dde.grad.jacobian(nut,X, i=0, j=0)
        nuty = dde.grad.jacobian(nut,X, i=0, j=1)
        ###################################################     
        eta = nu*tf.where(nutilde < 0.0,(1.0 + chi + 0.5*chi**2), (1.0 + chi))

        etax = dde.grad.jacobian(eta,X, i=0, j=0)
        etay = dde.grad.jacobian(eta,X, i=0, j=1)
        ###################################################
        vorZ = Vx - Uy
        Sdash = tf.math.sqrt(vorZ*vorZ)

        S = tf.math.sqrt(Sdash*Sdash + M*M) - M

        fv2 = 1 - chi/(1 + chi*fv1)
        ####################################################
        Sbar = nutilde*inv_dsqr*fv2/(kep*kep)
        
        Stilde = tf.math.maximum((1e-10)*tf.ones(shape=tf.shape(S),dtype = tf.dtypes.as_dtype(dde.config.default_float())), (S + Sbar))
        ###################################################
        rn = inv_dsqr*nutilde/(Stilde*kep*kep)

        r = tf_Stilde(rn)
        ###################################################
        g = r + cw2*(r**6 - r)

        fw = g*((1 + cw3**6)/((g**6) + cw3**6))**(1/6)
        ###################################################
        gn = 1 - (1000*chi**2)/(1+chi**2)
        ##################################################
        ##################################################
        sdiff = (1/sig)*(eta*(nutildexx + nutildeyy) + etax*nutildex + etay*nutildey)

        sp = tf.where(nutilde < 0.0, (cb1*S*nutilde*gn), (cb1*Stilde*nutilde))

        sd = tf.where(nutilde < 0.0, (cw1*(nutilde*nutilde)*inv_dsqr), (-cw1*fw*(nutilde*nutilde)*inv_dsqr))

        sc = (cb2/sig)*(nutildex*nutildex + nutildey*nutildey)
        #################################################
        return [sp, sd, sdiff, sc]
    return pde
########################################################################
@tf.function
def tf_Stilde(rn):
    """
    Evaluate value of r.

    :param rn: provisional value of r.
    :type rn: numpy or tf.Tensor
    :return: Final value of r capped between 0 and 10.
    :rtype: numpy or tf.Tensor

    """
    rdash = tf.where(rn < 0.0, tf_rlim(rn), rn)
    return tf_rtrue(rdash)

@tf.function   
def tf_rlim(rn):
    """
    Return limiting value of r.

    :param rn: provisional value of r.
    :type rn: numpy or tf.Tensor
    :return: Maximum value of r (10).
    :rtype: numpy or tf.Tensor

    """
    return 10.0*tf.ones(shape=tf.shape(rn),dtype = tf.dtypes.as_dtype(dde.config.default_float()))

@tf.function
def tf_rtrue(rn):
    """
    Return the minumimum between r and limiting value of r.

    :param rn: provisional value of r.
    :type rn: numpy or tf.Tensor
    :return: Minimum value of r between pn and limiting value.
    :rtype: numpy or tf.Tensor

    """
    return tf.math.minimum(rn,tf_rlim(rn))
