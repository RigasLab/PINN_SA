# PINN-SA
This repository contains the code used for the PINNs in the following paper:

```
Patel, Y., Mons, V., Marquet, O., Rigas, G. (2024).
Turbulence model augmented physics-informed neural networks for mean-flow reconstruction.
Physical Review Fluids.
```

This code implements a PINN to reconstruct a turbulent mean flow. In particular, it enables the use of a turbulence model augmented PINNs.

## Getting Started

This code is developed based on the DeepXDE package from
```
Lu, L. and Meng, X. and Mao, Z. and Karniadakis, G. E. (2021).
DeepXDE: A deep learning library for solving differential equations.
SIAM Review.
```

The main PINN training code is **runPINN_PH.py**.

The main config file used to setup the case is found in **PH_temp_config.json**

The training data is found in the folder **InputData/** and can be found from the following database
```
https://github.com/xiaoh/para-database-for-PIML
```

## Package installation in python

To run this code, the DeepXDE package must be installed:

```
pip install deepxde==1.9.1
```

Several backends such as TensorFlow, Pytorch, JAX etc can be used with DeepXDE. In this work, TensorFlow 1.x was used:
```
pip install tensorflow==2.12.0
```

Several other packages are required for DeepXDE and can be found at https://deepxde.readthedocs.io/en/latest/index.html


## Running the code

To run the code, use the following:
```
python runPINN_PH.py PH_temp_config.json
```

## Config file
To change and set up how the PINN is trained, only the config file needs to be adjusted. The key parameters are described below.

The included config files **BSL.json, SA.json** are used to generate the results for the PINN-DA-Baseline and PINN-DA-SA results for the above paper.

To select the training sequence, **[doPreTrain,doAdam, doLBFGSB]**:
```
"train_steps": [bool, bool, bool]
```


To change the number of training iterations in each sequence, **[preTrainSteps,AdamIterations,LBFGSBIterations,convergenceWindow]**:
```
"iter_list": [int, int, int, int]
```


To change the model hyperparameters, **[learning_rate, nodes/layer, layers, activation function, weight_initialiser]**:
```
"param": [float, int, int, str, str]
```


To set the number of **[PDE_collocation_pts, BC_collocation_pts, point_distribution]**:
```
"colloc": [int, int, str]
```

To set the PDE, one must select between **"HD"** (Helmholtz decomposition formulation) and **"RSTE"** (Reynolds stress formulation
```
"PDE": [int, int, str]
```


To activate the turbulence model, select **true**
```
"doSA": [bool, float]
```

**weight_pde, weight_reg, weight_data, etc.** define the cost function weights eg:
```
"weight_pde": [float, float, float, float, float]
```

The data resolution can be selected from **0p3,0p4,0p5,0p6,1p0** using:
```
"resolution": "0p5"
```
