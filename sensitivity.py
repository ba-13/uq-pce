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
    ModelOpts = {'Type':'Model','ModelFun':'uq_pce.uqlab.blackscholes_vectorized.put_option_price'}
else:
    raise NotImplementedError('Select model type: PUT or CALL')

Model = uq.createModel(ModelOpts)

Inputs = {
    "Marginals": [
        {"Name":"s",
         "Type":"Gaussian",
         "Parameters":[196,0.6]},
        {"Name":"t",
         "Type":"Gaussian",
         "Parameters":[0,1e-6]},
        {"Name":"T",
         "Type":"Uniform",
         "Parameters":[2,2.15]},
        {"Name":"K",
         "Type":"Uniform",
         "Parameters":[180,215]},
        {"Name":"sigma",
         "Type":"Uniform",
         "Parameters":[0.2,0.8]},
        {"Name":"r",
         "Type":"Gaussian",
         "Parameters":[0.0427,0.002]}
    ]
}

Input = uq.createInput(Inputs)

## input/output correlation
CorrSensOpts = {
    "Type":"Sensitivity",
    "Method":"Correlation",
    "Correlation": {
        "SampleSize":1e4
    }
}


##standard regression coefficients
SRCSensOpts = {
    'Type': 'Sensitivity',
    'Method': 'SRC',
    'SRC': {
        'SampleSize': 1e4
    }
}


##perturbation method
PerturbationSensOpts = {
    'Type': 'Sensitivity',
    'Method': 'Perturbation'
}


##cotter measure
CotterSensOpts = {
    'Type': 'Sensitivity',
    'Method': 'Cotter'
}


##morris indices
MorrisSensOpts = {
    "Type": "Sensitivity",
    "Method": "Morris"
}

MorrisSensOpts["Factors"] = {
    "Boundaries": 0.5
}

MorrisSensOpts["Morris"] = {
    "Cost": 1e4,
    "FactorSamples":1e4
}


##borgonovo indices
BorgonovoOpts = {
    'Type': 'Sensitivity',
    'Method': 'Borgonovo',
    'Borgonovo': {
        'SampleSize': 1e4
    }
}

SobolSensOpts = {
    'Type': 'Sensitivity',
    'Method': 'Sobol',
    'Sobol': {
        'SampleSize': 1e4
    }
}

ANCOVAOpts = {
    'Type': 'Sensitivity',
    'Method': 'ANCOVA',
    'ANCOVA': {
        'SampleSize': 1e4
    }
}

KucherenkoSensOpts = {
    'Type': 'Sensitivity',
    'Method': 'Kucherenko',
    'Kucherenko': {
        'SampleSize': 1e4
    }
}

#CorrAnalysis = uq.createAnalysis(CorrSensOpts)
#uq.print(CorrAnalysis)
#uq.display(CorrAnalysis)

#SRCAnalysis = uq.createAnalysis(SRCSensOpts)
#uq.print(SRCAnalysis)

#PerturbationAnalysis = uq.createAnalysis(PerturbationSensOpts)
#uq.print(PerturbationAnalysis)

#CotterAnalysis = uq.createAnalysis(CotterSensOpts)
#uq.print(CotterAnalysis)

#MorrisAnalysis = uq.createAnalysis(MorrisSensOpts)
#uq.print(MorrisAnalysis)
#uq.display(MorrisAnalysis);

#BorgonovoAnalysis = uq.createAnalysis(BorgonovoOpts)
#uq.print(BorgonovoAnalysis)

#SobolAnalysis = uq.createAnalysis(SobolSensOpts)
#uq.print(SobolAnalysis)

#ANCOVAAnalysis = uq.createAnalysis(ANCOVAOpts)
#uq.print(ANCOVAAnalysis)

#KucherenkoAnalysis = uq.createAnalysis(KucherenkoSensOpts)
#uq.print(KucherenkoAnalysis)

Session.quit()