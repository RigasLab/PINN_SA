.. PINN_SA documentation master file, created by
   sphinx-quickstart on Mon Apr  3 18:51:23 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PINN_SA's documentation!
===================================
This directory presents an implementation of DeepXDE package which allows changing problem setup from a single config file. This allows the case, hyperparameters and other parameters to be changed with a single file. This code was used for the work XXXX, which solves the inverse problem for the mean flow RANS equations.

**Documentation**: xxx

.. image:: images/pinn.png

Features
--------
This code has the ability to chnage many parameters such as:
- changing architecture (nodes and layers), activation functions and learning rate
- changing the weights of individual loss components
- changing the PDE for the problem
- changing the individual case
- changing number of iterations and convergence criteria


Check out the :doc:`usage` section for further information, including how to
:ref:`install <installation>` the project.

User Guide
----------

.. toctree::
   :maxdepth: 2
   :caption: Installation Guide:
   
   usage
 
Examples
---------

.. toctree::
   :maxdepth: 1
   :caption: Example Cases:
   
   demos/examples.PH
   
API Reference
----------

.. toctree::
   :maxdepth: 2
   :caption: API: 
   
   api
   modules/src.PINNBox.BaseFunctions
   modules/src.PINNBox.PINNFunctions
   modules/src.PINNBox.PINNPlotter
   modules/src.PINNBox.PDEBox
   modules/src.PINNBox.CaseBox
   modules/src.PINNBox.BC
   modules/src.PeriodicHills



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
