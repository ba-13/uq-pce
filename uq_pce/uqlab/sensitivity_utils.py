from uqpylab import sessions, display_general, display_util
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from enum import Enum
from scipy import stats


def lra_sobol(uq,degree):
    LRASobol={"Type":"Sensitivity","Method":"Sobol",
        "Sobol":{"Order":degree}}
    LRA_SA = uq.createAnalysis(LRASobol)
    uq.print(LRA_SA)
    uq.display(LRA_SA)

def pce_sobol(uq,degree,PCE):
    PCE = PCE
    PCESobol={"Type":"Sensitivity","Method":"Sobol",
        "Sobol":{"Order":degree}}
    PCE_SA = uq.createAnalysis(PCESobol)
    uq.print(PCE_SA)

def corr_sens(uq,N,Input):
    Input = Input
    CorrSensOpts = {"Type":"Sensitivity","Method":"Correlation",
    "Correlation":{"SampleSize":N}}
    CorrAnalysis = uq.createAnalysis(CorrSensOpts)
    uq.print(CorrAnalysis)
    uq.display(CorrAnalysis)

def src_sens(uq,N,Input):
    Input=Input
    SRCSensOpts = {'Type': 'Sensitivity',
    'Method': 'SRC','SRC': {'SampleSize': N}}
    SRCAnalysis = uq.createAnalysis(SRCSensOpts)
    uq.print(SRCAnalysis)

def perturbation(uq,Input):
    Input = Input
    PerturbationSensOpts = {'Type': 'Sensitivity',
    'Method': 'Perturbation'}
    PerturbationSensOpts['Gradient']={'Method':'Centered'}
    PerturbationAnalysis = uq.createAnalysis(PerturbationSensOpts)
    uq.print(PerturbationAnalysis)

def cotter(uq):
    CotterSensOpts = {'Type': 'Sensitivity',
    'Method': 'Cotter'}
    CotterAnalysis = uq.createAnalysis(CotterSensOpts)
    uq.print(CotterAnalysis)
    uq.display(CotterAnalysis);

def morris(uq,cost,factorsamples):
    MorrisSensOpts = {"Type": "Sensitivity","Method": "Morris"}
    MorrisSensOpts["Morris"] = {"Cost": cost,"FactorSamples":factorsamples}
    MorrisAnalysis = uq.createAnalysis(MorrisSensOpts)
    uq.print(MorrisAnalysis)
    uq.display(MorrisAnalysis);

def borgonovo(uq,N):
    BorgonovoOpts = {'Type': 'Sensitivity',
    'Method': 'Borgonovo','Borgonovo': {
        'SampleSize': N}}
    BorgonovoAnalysis = uq.createAnalysis(BorgonovoOpts)
    uq.print(BorgonovoAnalysis)

def sobol_sens(uq,N,degree,sampling):
    SobolSensOpts = {'Type': 'Sensitivity',
    'Method': 'Sobol','Sobol': {
        'SampleSize': N,'Order':degree}}
    SobolAnalysis = uq.createAnalysis(SobolSensOpts)
    SobolSensOpts['Sobol']['Sampling'] = sampling
    uq.print(SobolAnalysis)

def ancova_sens(uq,N):
    ANCOVAOpts = {'Type': 'Sensitivity',
    'Method': 'ANCOVA','ANCOVA': {
        'SampleSize': N}}
    ANCOVAAnalysis = uq.createAnalysis(ANCOVAOpts)
    uq.print(ANCOVAAnalysis)


class SensType(Enum):
    LRA = lra_sobol
    PCE = pce_sobol
    CORR = corr_sens
    SRC = src_sens
    PA = perturbation
    COT = cotter
    MOR = morris
    BOR = borgonovo
    SOBOL = sobol_sens
    ANCOVA = ancova_sens