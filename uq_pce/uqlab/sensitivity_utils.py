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

def pce_sobol(uq,degree):
    PCESobol={"Type":"Sensitivity","Method":"Sobol",
        "Sobol":{"Order":degree}}
    PCE_SA = uq.createAnalysis(PCESobol)
    uq.print(PCE_SA)
    uq.display(PCE_SA)

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
    PerturbationAnalysis = uq.createAnalysis(PerturbationSensOpts)
    uq.print(PerturbationAnalysis)

def cotter(uq,Input):
    Input = Input
    CotterSensOpts = {'Type': 'Sensitivity',
    'Method': 'Cotter'}
    CotterAnalysis = uq.createAnalysis(CotterSensOpts)
    uq.print(CotterAnalysis)

def morris(uq,Input,boundaries,cost,factorsamples):
    MorrisSensOpts = {"Type": "Sensitivity","Method": "Morris"}
    MorrisSensOpts["Factors"] = {"Boundaries": boundaries}
    MorrisSensOpts["Morris"] = {"Cost": cost,"FactorSamples":factorsamples}
    MorrisAnalysis = uq.createAnalysis(MorrisSensOpts)
    uq.print(MorrisAnalysis)
    uq.display(MorrisAnalysis);

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