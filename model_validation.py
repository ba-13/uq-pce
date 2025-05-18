from uqpylab import sessions, display_general, display_util
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
#from test import train_validate_loop
#from uq_pce.model.model import get_dataset, INPUT_DISTRIBUTION, ModelType, NUM_VARIABLES
from uq_pce.pce.pce import compute_psi, pce_get_coefficients
from uq_pce.uqlab.base import (init_uqlab, call_option_price,
                                put_option_price, ModelType,
                                model_inputs, create_pce,
                                eval_pce, create_lra)

Instance = 'https://uqcloud.ethz.ch'
Token = '54670830aaf970cdc8b7dd7e9015d308665f9096'

Session  = sessions.cloud(host=Instance,token=Token)
np.random.seed(42)
degree = 3

uq = init_uqlab(Session=Session,seed=42)
Model, _ = model_inputs(uq,ModelType.CALL)
PCE, _ = create_pce(uq,130,'LHS','OLS',Model,degree) #methods: 'OLS','OMP','LARS'
Yval,YPCE = eval_pce(uq,1e4,Model,PCE)
print(PCE['Error']['LOO'])

##model variables
uq.display(PCE)
coeffs_uqlab = np.array(PCE['PCE']['Coefficients'],dtype=float)
coeffs_model = pce_get_coefficients(130,degree)
print(coeffs_uqlab,coeffs_model)
delta = coeffs_uqlab - coeffs_model
L2_err      = np.linalg.norm(delta)
rel_L2_err  = L2_err / np.linalg.norm(coeffs_model)
Linf_err    = np.max(np.abs(delta))
print(f"L2: {L2_err}, rel_err: {rel_L2_err}, Linf_err: {Linf_err}")
mu_test  = coeffs_model[0]
mu_ctrl  = coeffs_uqlab[0]
var_test = np.sum(coeffs_model[1:]**2)
var_ctrl = np.sum(coeffs_uqlab[1:]**2)
print(mu_test,mu_ctrl,var_test,var_ctrl)

##TODO: 
# add error metrics from uqlab model
# compare moments

plt.figure(figsize=(8,8))
plt.scatter(Yval,YPCE,alpha=0.5)
plt.plot([np.min(Yval), np.max(Yval)], [np.min(Yval), np.max(Yval)], 'r--')
plt.title('Test PCE vs. Real')
plt.xlabel('True Option Prices (Y)')
plt.ylabel('Surrogate Prices')
plt.grid(True)
plt.tick_params(axis='both')
plt.show()



##low-rank approximation
LRAOpts = {
    "Type":"Metamodel",
    "MetaType":"LRA",
    "Input":Input["Name"],
    "FullModel":Model["Name"],
    "Rank":list(range(1,11)),
    "Degree":2,
    "ExpDesign":{
        "NSamples":40
    }
}
LRA = uq.createModel(LRAOpts)

##sensitivity analysis - LRA
LRASobol = {
    "Type":"Sensitivity",
    "Method":"Sobol",
    "Sobol": {
        "Order": 3
    }
}
LRASobolAnalysis = uq.createAnalysis(LRASobol)
uq.print(LRASobolAnalysis)

##sensitivity analysis - PCE
PCESobol = {
    "Type":"Sensitivity",
    "Method":"Sobol",
    "Sobol": {
        "Order":3
    }
}
PCESobolAnalysis = uq.createAnalysis(PCESobol)
uq.print(PCESobolAnalysis)

Session.quit()