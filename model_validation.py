from uqpylab import sessions, display_general, display_util
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
#from test import train_validate_loop
from uq_pce.model.model import INPUT_DISTRIBUTION, ModelType, NUM_VARIABLES,get_dataset_val
from uq_pce.pce.pce import compute_psi, pce_get_coefficients
from uq_pce.uqlab.base import (init_uqlab, call_option_price,
                                put_option_price, ModelTypeUQL,
                                model_inputs, create_pce,
                                eval_pce, create_lra)

Instance = 'https://uqcloud.ethz.ch'
Token = '54670830aaf970cdc8b7dd7e9015d308665f9096'

Session  = sessions.cloud(host=Instance,token=Token)
np.random.seed(42)
degree = 3

uq = init_uqlab(Session=Session,seed=42)
Model, Input = model_inputs(uq,ModelTypeUQL.CALL)
PCE, _ = create_pce(uq,130,'LHS','OLS',Model,degree) #methods: 'OLS','OMP','LARS'
Yval,YPCE = eval_pce(uq,1e4,Model,PCE)
print(PCE['Error']['LOO'])

##model variables
uq.display(PCE)
coeffs_uqlab = np.array(PCE['PCE']['Coefficients'],dtype=float)
coeffs_model = pce_get_coefficients(130,degree)
alphas_uqlab = np.array(PCE['PCE']['Basis']['Indices']) 
Xtest = uq.getSample(N=10000)
Xtest_filtered = np.delete(Xtest, 1, axis=1)
Y_UQL = uq.evalModel(PCE,Xtest)
Y_Model= uq.evalModel(Model,Xtest)
Y_test, _,idx_attr_map = get_dataset_val(sample=Xtest_filtered,model=ModelType.CALL)
_,alphas_model = compute_psi(Xtest_filtered, 3, idx_attr_map)
print(PCE['Error'])

uqlab_idx = np.lexsort(alphas_uqlab.T[::-1])
model_idx = np.lexsort(alphas_model.T[::-1])
idx_u, val_u = alphas_uqlab[uqlab_idx], coeffs_uqlab[uqlab_idx]
idx_m, val_m = alphas_model[model_idx], coeffs_model[model_idx]
assert np.all(idx_u == idx_m)
rmse_models = np.sqrt(np.mean((Y_UQL-Y_test)**2))

delta = val_u - val_m
L2_err      = np.linalg.norm(delta)
rel_L2_err  = L2_err / np.linalg.norm(val_m)
Linf_err    = np.max(np.abs(delta))
rcorr, _       = pearsonr(val_u, val_m)
r2         = r2_score(val_u, val_m)
#mu_test  = coeffs_model[0]
#mu_ctrl  = coeffs_uqlab[0]
#var_test = np.sum(coeffs_model[1:]**2)
#var_ctrl = np.sum(coeffs_uqlab[1:]**2)
print(f'RMSE: {rmse_models}, Coefficients_L2: {L2_err}, Relative_L2: {rel_L2_err}, Linf_err: {Linf_err}, Pearson R: {rcorr}, R^2={r2}')
#print(mu_test,mu_ctrl,var_test,var_ctrl)

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

plt.figure(figsize=(8,8))
plt.scatter(Y_test,Y_UQL,alpha=0.5)
plt.plot([np.min(Yval), np.max(Yval)], [np.min(Yval), np.max(Yval)], 'r--')
plt.title('Test UQpyLab vs. Custom Model')
plt.xlabel('UQpyLab Surrogate')
plt.ylabel('Custom Surrogate')
plt.grid(True)
plt.tick_params(axis='both')
plt.show()

LRA = create_lra(uq,1e4,Model,Input,degree)

Session.quit()