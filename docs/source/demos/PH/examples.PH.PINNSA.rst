PINN-SA with Turbulence Model Augmentation
=============================================

Problem Setup
--------------

We will solve the inverse problem with the Reynolds-Averaged Navier-Stokes (RANS) equations:

.. math:: \frac{\partial U_{i}}{\partial x_{i}} = 0,
 
.. math:: U_{j}\frac{\partial U_{i}}{\partial x_{j}} + \frac{1}{\rho}\frac{\partial P}{\partial x_{j}} + \frac{\partial (2\nu S_{ij})}{\partial x_{j}} + f_{p,i}.

This will be solved across the Periodic Hill domain at :math:`Re=5600`. The full case description can be found in XXX.

Implementaton
--------------

To setup this case we must make the following modifications to the underlying config file from XXX.

First we must set the case to the periodic hill config and give the test folder a name (``RST-E``)

.. code-block:: json
	
	"case": "PH",
	"test_name": "RST-E",
	
The next step is to define the training setup. This includes number of iterations, neural network architecture.

.. code-block:: json
	
	"iter_list": [10000,500,500,200000],
	"train_steps": [false, true, true],
	"param": [5e-4, 5, 2, "tanh", "Glorot uniform"],

The PDE must be now defined as well as the collocation points over which to evaluate it.

.. code-block:: json

	"colloc": [100, 100, "Hammersley"],
	"PDE"  : "RSTE",
	"fixedfp": [false, -0.0110],
	"doSA": [false, 1e-10],
	"hardConstraint": false,

Select the weights for the PDE and boundary condition losses.

.. code-block:: json  

	"weight_pde": [1,1,1,1,1],
	"weight_reg": [1e-2],
	"weight_data": [10,10],
	"weight_wall": [2.5,2.5,10,10,10],
	"weight_periodic": [1,1,1,1,1,1],
	"weight_massflow": [1e-1],
	
To select the data resolution, use the following:

.. code-block:: json

	"resolution": "0p5"

For RST-E, both mean flow data and Reynolds stress data will be used to train the model.

Complete Config
---------------------

.. literalinclude:: ../../../../examples/PeriodicHill/PH_temp_config.json
  :language: json