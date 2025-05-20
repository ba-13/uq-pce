from uqpylab import sessions, display_general, display_util
import numpy as np
import matplotlib.pyplot as plt
from uq_pce.uqlab.base import (init_uqlab, call_option_price, 
                               put_option_price, ModelTypeUQL, 
                               model_inputs, create_pce, 
                               eval_pce, create_lra, kriging)

Instance = 'https://uqcloud.ethz.ch'
Token = '54670830aaf970cdc8b7dd7e9015d308665f9096'

Session  = sessions.cloud(host=Instance,token=Token)
degree = 3

uq = init_uqlab(Session=Session,seed=42)
Model, Input = model_inputs(uq,ModelTypeUQL.CALL)
PCE, _ = create_pce(uq,130,'LHS','OLS',Model,degree)
LRA = create_lra(uq,1e4,Model,Input,degree)
N = 1e4

X = uq.getSample(N=N)
Y = uq.evalModel(Model,X)

KrigingMatern = kriging(uq, Model, N,'PCK', 'Sequential', 'Gaussian', 'LHS', 'LARS', 130, 3) 
KrigingLinear = kriging(uq, Model, N,'PCK', 'Sequential', 'Linear', 'LHS', 'LARS', 130, 3) 
KrigingExp = kriging(uq, Model, N,'PCK', 'Sequential', 'Exponential', 'LHS', 'LARS', 130, 3) 
#available metatypes: 'PCK', 'Kriging'
#available modes: 'Sequential', 'Optimal'
#available solvers: 'OLS', 'LARS', 'OMP'
#available correlation families: 'Gaussian', 'Linear', 'Exponential'

Xval = uq.getSample(N=1e4, Method='LHS')
Yval = uq.evalModel(Model, Xval)

[YMeanMat,YVarMat] = uq.evalModel(KrigingMatern, Xval, nargout=2)
[YMeanLin,YVarLin] = uq.evalModel(KrigingLinear, Xval, nargout=2)
[YMeanExp,YVarExp] = uq.evalModel(KrigingExp, Xval, nargout=2)

plt.plot(Xval, YMeanMat,'-')
#plt.plot(Xval, YMeanLin,'-')
#plt.plot(Xval, YMeanExp, '--')
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