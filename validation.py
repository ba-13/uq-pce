from uqpylab import sessions, display_general, display_util
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
#from uq_pce.model import blackscholes
#from uq_pce.uqlab.base import (init_uqlab, call_option_price,
#                                put_option_price, ModelType,
#                                model_inputs, create_pce,
#                                eval_pce, create_lra)

Token = '54670830aaf970cdc8b7dd7e9015d308665f9096'
Instance = 'https://uqcloud.ethz.ch'

Session  = sessions.cloud(host=Instance,token=Token)
uq = Session.cli
Session.reset()

seed = 42
uq.rng(seed,'twister');

ModelType = 'CALL'

if ModelType == 'CALL':
    ModelOpts = {'Type':'Model','ModelFun':'uq_pce.uqlab.blackscholes_vectorized.call_option_price'}
elif ModelType == 'PUT':
    ModelOpts = {'Type':'Model','ModelFun':'uq_pce.model.blackscholes_vectorized.put_option_price'}
else:
    raise NotImplementedError('Select model type: PUT or CALL')

Model = uq.createModel(ModelOpts)

Inputs = {
    'Marginals': [
        {'Type':'Gaussian',
         'Parameters':[196,0.6]},
        {'Type':'Gaussian',
         'Parameters':[0,0]},
        {'Type':'Uniform',
         'Parameters':[2,2.15]},
        {'Type':'Uniform',
         'Parameters':[180,215]},
        {'Type':'Uniform',
         'Parameters':[0.2,0.8]},
        {'Type':'Gaussian',
         'Parameters':[0.0427,0.03]}
    ]
}

Input = uq.createInput(Inputs)

MetaOpts = {
    'Type':'Metamodel',
    'MetaType':'PCE',
    'Method':'OLS'
    }
MetaOpts['FullModel'] = Model['Name']
MetaOpts['Degree'] = 3
Xval = uq.getSample(N=1e4)
Yval = uq.evalModel(Model,Xval)


MetaOpts['ExpDesign'] = {
    'NSamples':130,
    'Sampling':'LHS'
}

PCE_LHS40 = uq.createModel(MetaOpts)
uq.print(PCE_LHS40)
Y_LHS40 = uq.evalModel(PCE_LHS40,Xval)
uq.display(PCE_LHS40)

##model variables
print(PCE_LHS40['PCE']['Coefficients'])

plt.figure(figsize=(8,8))
plt.scatter(Yval, Y_LHS40,alpha=0.5)
plt.plot([np.min(Yval), np.max(Yval)], [np.min(Yval), np.max(Yval)], 'r--')
plt.title('Test PCE vs. Real')
plt.xlabel('True Option Prices (Y)')
plt.ylabel('Surrogate Prices')
plt.grid(True)
plt.tick_params(axis='both')
#plt.xlim(np.min(Yval), np.max(Yval))
#plt.ylim(np.min(Yval), np.max(Yval))
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
        "NSamples":130
    }
}
LRA = uq.createModel(LRAOpts)

##sensitivity analysis - LRA
LRASobol = {
    "Type":"Sensitivity",
    "Method":"Sobol",
    "Sobol": {
        "Order": 2
    }
}
LRASobolAnalysis = uq.createAnalysis(LRASobol)
uq.print(LRASobolAnalysis)

##sensitivity analysis - PCE
PCESobol = {
    "Type":"Sensitivity",
    "Method":"Sobol",
    "Sobol": {
        "Order":2
    }
}
PCESobolAnalysis = uq.createAnalysis(PCESobol)
uq.print(PCESobolAnalysis)

Session.quit()