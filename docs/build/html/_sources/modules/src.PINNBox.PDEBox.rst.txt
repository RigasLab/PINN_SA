PDEBox
=======
Folder where all PDE equations for use with PINNs are placed
.. _PDEbox:

Reynolds-Averaged Navier Stokes(RANS) equations
--------------------------------------------------
The RANS equations are used to solve for mean flow. Several formulations are applied to account for the unclosed nature:
 
The RST-E formulation is the most basic fromulation of the RANS equations and is defined as:

.. math:: \frac{\partial U_{i}}{\partial x_{i}} = 0,
 
.. math:: U_{j}\frac{\partial U_{i}}{\partial x_{j}} + \frac{1}{\rho}\frac{\partial P}{\partial x_{j}} + \frac{\partial (2\nu S_{ij})}{\partial x_{j}} + f_{p,i},
 
where the mean strain rate tensor is defined as:

.. math:: S_{ij} = \frac{1}{2}(\frac{\partial U_{i}}{\partial x_{j}} + \frac{\partial U_{j}}{\partial x_{i}})
 
.. autofunction:: src.PINNBox.PDEBox.RANS.RANS_B_RSTE

The HD formulation (without turbulence modelling) splits the Reynolds forcing (divergence of Reynolds stress) into a potential and solenoidal part such that the RANS equations now appear as:

.. math:: \frac{\partial U_{i}}{\partial x_{i}} = 0,
 
.. math:: U_{j}\frac{\partial U_{i}}{\partial x_{j}} + \frac{1}{\rho}\frac{\partial (P - \phi)}{\partial x_{j}} + \frac{\partial (2\nu S_{ij})}{\partial x_{j}} + f_{s,i} + f_{p,i},
 
.. math:: \frac{\partial f_{s,i}}{\partial x_{i}} = 0,
 
.. autofunction:: src.PINNBox.PDEBox.RANS.RANS_B_HD

One can also augment the RANS equations with a turbulence model:

.. math:: \frac{\partial U_{i}}{\partial x_{i}} = 0,
 
.. math:: U_{j}\frac{\partial U_{i}}{\partial x_{j}} + \frac{1}{\rho}\frac{\partial (P - \phi)}{\partial x_{j}} + \frac{\partial 2(\nu + \nu_{t}) S_{ij}}{\partial x_{j}} + f_{s,i} + f_{p,i},
 
.. math:: \frac{\partial f_{s,i}}{\partial x_{i}} = 0,
 
.. math:: U_{j}\frac{\partial \tilde{\nu}}{\partial x_{j}} + S_{p} + S_{d} + S_{c} + S_{diff} = 0.

.. autofunction:: src.PINNBox.PDEBox.RANS.RANS_SA_HD