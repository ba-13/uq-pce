from uqpylab import sessions, display_general, display_util
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from enum import Enum
from scipy import stats

Instance = 'https://uqcloud.ethz.ch'
Token = '54670830aaf970cdc8b7dd7e9015d308665f9096'

Session  = sessions.cloud(host=Instance,token=Token)

def init_uqlab(Session, seed):
    uq = Session.cli
    Session.reset()
    uq.rng(seed,'twister');
    return uq

#Session.quit()


def call_option_price(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Args
    --------
        s (npt.NDArray[np.float64]): asset price
        t (npt.NDArray[np.float64]): prediction time
        T (npt.NDArray[np.float64]): maturity time
        K (npt.NDArray[np.float64]): strike price
        sigma (npt.NDArray[np.float64]): volatility, stddev of asset price
        r (npt.NDArray[np.float64]): risk-free interest rate
    Returns
    --------
        npt.NDArray[np.float64]: call option price
    """
    s     = X[..., 0]
    t     = X[..., 1]
    T     = X[..., 2]
    K     = X[..., 3]
    sigma = X[..., 4]
    r     = X[..., 5]

    tau = np.maximum(T - t, 0.0)
    sqrt_tau = np.sqrt(tau, where=(tau>0), out=np.zeros_like(tau))

    d1 = (np.log(s / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    return s * stats.norm.cdf(d1) - K * np.exp(-r * tau) * stats.norm.cdf(d2)

def put_option_price(X: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Args
    --------
        s (npt.NDArray[np.float64]): asset price
        t (npt.NDArray[np.float64]): prediction time
        T (npt.NDArray[np.float64]): maturity time
        K (npt.NDArray[np.float64]): strike price
        sigma (npt.NDArray[np.float64]): volatility
        r (npt.NDArray[np.float64]): risk-free interest rate
    Returns
    --------
        npt.NDArray[np.float64]: put option price
    """
    s     = X[..., 0]
    t     = X[..., 1]
    T     = X[..., 2]
    K     = X[..., 3]
    sigma = X[..., 4]
    r     = X[..., 5]
    
    tau = np.maximum(T - t, 0.0)
    sqrt_tau = np.sqrt(tau, where=(tau>0), out=np.zeros_like(tau))

    d1 = (np.log(s / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    return K * np.exp(-r * tau) * stats.norm.cdf(-d2) - s * stats.norm.cdf(-d1)

class ModelType(Enum):
    CALL = call_option_price
    PUT = put_option_price

model = ModelType.CALL

INPUT_DISTRIBUTION = [
    {"Name":"s","Type":"Gaussian","Parameters":[196, 0.6]},
    {"Name":"t","Type":"Gaussian","Parameters":[0, 0]},
    {"Name":"T","Type":"Uniform","Parameters":[2, 2.25]},
    {"Name":"K", "Type":"Uniform","Parameters":[180, 215]},
    {"Name":"sigma","Type":"Uniform","Parameters":[0.2, 0.8]},
    {"Name":"r","Type":"Gaussian","Parameters":[0.0427, 0.03]},
    ]

def model_inputs(uq,Model):
    modelname = "uq_pce.uqlab.base."+str(model).split('function ')[1].split(' at')[0]
    ModelOpts = {"Type":"Model","ModelFun":modelname}
    Model = uq.createModel(ModelOpts)
    Inputs = {"Marginals": INPUT_DISTRIBUTION}
    Input = uq.createInput(Inputs)
    return [Model, Input]

def create_pce(uq,expN,Sampling, Method,Model,degree):
    MetaOpts = {'Type':'Metamodel','MetaType':'PCE','Method':Method}
    MetaOpts['FullModel'] = Model['Name']
    MetaOpts['Degree'] = degree
    MetaOpts['ExpDesign'] = {'NSamples':expN, 'Sampling': Sampling}
    PCE = uq.createModel(MetaOpts)
    PCESobol = {"Type":"Sensitivity","Method":"Sobol","Sobol":{"Order":degree}}
    PCESobolAnalysis = uq.createAnalysis(PCESobol)
    return [PCE, PCESobolAnalysis]

def eval_pce(uq,expN,Model,PCEModel):
    Xval = uq.getSample(N=expN)
    Yval = uq.evalModel(Model,Xval)
    YPCE = uq.evalModel(PCEModel,Xval)
    return Yval, YPCE

def create_lra(uq, expN, Model, Input, degree):
    LRAOpts = {
        "Type":"Metamodel",
        "MetaType":"LRA",
        "Input":Input["Name"],
        "FullModel":Model["Name"],
        "Rank":list(range(1,11)),
        "Degree":degree,
        "ExpDesign":{"NSamples":expN}
    }
    LRA = uq.createModel(LRAOpts)
    return LRA