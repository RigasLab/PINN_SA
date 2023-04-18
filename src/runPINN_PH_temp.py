import os
import sys
import json


#CaseLoader class
#caseloader.init - define geometry and BC, calc nu+rho, get BC list
#caseloader.getPDE (get pde and set netshape, get PDEnames)
#caseLoader.getNetShape (get netshape)
#caseLoader.getOutputTransform (get transform)
#caseLoader.getData - loadData and sample resolution

[pde, netShape] = pde_loader("RANS", testcfg['PDE'], pde_other=[rho,nu,fp,doSA])



##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
config_path = sys.argv[1]
with open(config_path, "r") as read_file:
    testcfg = json.load(read_file)

if(testcfg['backend'] == 'v2'):
    os.environ["DDE_BACKEND"] = "tensorflow"
elif(testcfg['backend'] == 'v1'):
    os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"


###################################################################
import deepxde as dde
from deepxde.backend import tf

import numpy as np
import matplotlib.pyplot as plt

from PINNBox.CaseBox.CaseLoader import initCase, postProTrain, getPlotter

from timeit import default_timer as timer
start = timer()
print("hello")
####################################################################################################################
####################################################################################################################
os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/apps/cuda/11.2.2/"

dde.config.set_default_float("float64")

#dde.backend.tf.logging.set_verbosity(20)
#print(dde.backend.tf.logging.get_verbosity())
####################################################################################################################
####################################################################################################################
case = testcfg['case']

##*********************
test_name = testcfg['test_name'] + "/"

iter_pre, iter, iter2, pat = testcfg['iter_list']
lr, nodes, layers, activation, initialiser = testcfg['param']
N_train, N_boundary, dist_type = testcfg['colloc']

pde_type = testcfg['PDE'] #RSTE,HD

fixedfp = testcfg['fixedfp'][0]
if (fixedfp == False):
    fp = dde.Variable(testcfg['fixedfp'][1], dtype=dde.config.default_float())
    testcfg['fixedfp'][1] = fp

other = [testcfg['fixedfp'],testcfg['doSA'],testcfg['massflow']]

weights = [testcfg['weight_pde'],testcfg['weight_reg'],
           testcfg['weight_data'],testcfg['weight_wall'],
           testcfg['weight_periodic'],testcfg['weight_massflow']]

doPretrain, doAdam, doLBFGS = testcfg["train_steps"]

NN_type = "FNN"
hard_constraint = testcfg["hardConstraint"]

if(testcfg['doSA'][0] == False):
    hard_constraint = False

isSol      = testcfg["isSol"]

loadModel = testcfg["loadModel"]
getAnchor = testcfg["loadColloc"]

resolution = testcfg["resolution"]
###############################################################################
##################################################################################
##################################################################################
[domain, bc, loss_name, loss_weights, transform, func_sol, X_anchor, path_list, aux_D] = initCase(case,getAnchor,test_name,other,weights,resolution)



print("LOADED")

final_path, path_M, path_A, path_B, path_C = path_list
##################################################################################
###############################################################################
##################################################################################
if(isSol == False):
    data = dde.data.PDE(
        domain,
        pde,
        bc,
        N_train,
        N_boundary,
        dist_type,
        auxiliary_var_function = aux_D
        )

    met_list = []
else:
    data = dde.data.PDE(
        domain,
        pde,
        bc,
        N_train,
        N_boundary,
        dist_type,
        solution = func_sol,
        auxiliary_var_function = aux_D
        )

    met_list = ["l2 relative error"]

print("CONFIGURED")

## Add Anchors
if(getAnchor):
    data.add_anchors(X_anchor)

#####################################################################################
plt.plot(data.train_x_bc[:,0], data.train_x_bc[:,1],'rx')
plt.plot(data.train_x_all[:,0], data.train_x_all[:,1],'g.')
np.save(final_path + "DataPoints.npy", data.train_x_all)
plt.savefig(final_path + "DataPoints.png")
plt.clf()
##################################################################################
#################################################################################
##################################################################################
numIn, numOut = netShape

layer_size = [numIn] + [nodes] * layers + [numOut]

if(NN_type == "FNN"):
    net = dde.nn.FNN(layer_size,activation,initialiser)
elif(NN_type == "PFNN"):
    net = dde.nn.PFNN(layer_size,activation,initialiser)

if(hard_constraint):
    net.apply_output_transform(transform)

##################################################################################
model = dde.Model(data,net)
##################################################################################
##################################################################################
##################################################################################
es = dde.callbacks.EarlyStopping(min_delta=0, patience=pat)

if(fixedfp == False):
    fp_track  = dde.callbacks.VariableValue([fp],period=100,filename=final_path + 'fp.dat')
    cb_list = [es, fp_track]
    cb_list_lbfgs = [fp_track]
else:
    cb_list = [es]
    cb_list_lbfgs = []  


####################################################################################
# Pre optimisation
if(doPretrain):
    dde.optimizers.config.set_LBFGS_options(maxiter=iter_pre)
    model.compile("L-BFGS-B", loss_weights=loss_weights, metrics=met_list)
    history_pre, train_state_pre = model.train()

    #postProTrain(history_pre, path_M, loss_name, "Pre", "/model-pre-last", model)

##################################################################################
# Adam optimisation
if(doAdam):
    model.compile("adam",lr=lr,loss_weights=loss_weights, metrics=met_list)
    if(loadModel[0]):
        history, train_state = model.train(iter,display_every=100,callbacks = cb_list, model_restore_path=loadModel[1])
    else:
        history, train_state = model.train(iter,display_every=100,callbacks = cb_list)#,external_trainable_variables=[fp])

    postProTrain(history, path_A, loss_name, "Adam", "/model-adam-last", model)

##################################################################################
# L-BFGS-B optimisation
if(doLBFGS):
    dde.optimizers.config.set_LBFGS_options(maxiter=iter2)
    model.compile("L-BFGS-B", loss_weights=loss_weights, metrics=met_list)#,external_trainable_variables=[fp])
    history_2, train_state_2 = model.train(callbacks = cb_list_lbfgs)

    postProTrain(history_2, path_B, loss_name, "L-BFGS", "/model-lbfgs-last", model)
##################################################################################
##################################################################################
##################################################################################
plot_dir = final_path

plot_func = getPlotter(case)

plot_func(plot_dir,func_sol,model,pde)

end = timer()
print(end - start)

