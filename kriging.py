from uqpylab import sessions, display_general, display_util
import numpy as np
import matplotlib.pyplot as plt

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
        {'Type':'Uniform',
         'Parameters':[0,0]},
        {'Type':'Uniform',
         'Parameters':[2,2]},
        {'Type':'Uniform',
         'Parameters':[180,215]},
        {'Type':'Uniform',
         'Parameters':[0.2,0.8]},
        {'Type':'Gaussian',
         'Parameters':[0.0427,0.002]}
    ]
}

Input = uq.createInput(Inputs)

X = uq.getSample(N=1e3)
Y = uq.evalModel(Model,X)
                 
MetaOpts = {
    "Type": "Metamodel",
    "MetaType": "Kriging"
}



MetaOpts["ExpDesign"] = {
    'NSamples':40,
    'Sampling':'LHS'
}

KrigingMatern = uq.createModel(MetaOpts)

uq.print(KrigingMatern)

MetaOpts['Corr'] = {
    "Family": "Linear"
}
KrigingLinear = uq.createModel(MetaOpts)

uq.print(KrigingLinear)

MetaOpts['Corr']['Family'] = 'Exponential'
KrigingExp = uq.createModel(MetaOpts)
uq.print(KrigingExp)

Xval = uq.getSample(N=1e3, Method='LHS')
Yval = uq.evalModel(Model, Xval)

[YMeanMat,YVarMat] = uq.evalModel(KrigingMatern, Xval, nargout=2)
[YMeanLin,YVarLin] = uq.evalModel(KrigingLinear, Xval, nargout=2)
[YMeanExp,YVarExp] = uq.evalModel(KrigingExp, Xval, nargout=2)

plt.plot(Xval, YMeanMat,'-')
plt.plot(Xval, YMeanLin,'-')
plt.plot(Xval, YMeanExp, '--')
#plt.plot(Xval, Yval, '-k')
plt.plot(X, Y, 'ko',markersize=2)

plt.xlim([170, 220])
#plt.ylim([-15, 20])
plt.legend(['Kriging, R: Matern 5/2', 
            'Kriging, R: Linear', 
            'Kriging, R: Exponential',
            'True model', 'Observations'],
          loc='upper left')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

Session.quit()