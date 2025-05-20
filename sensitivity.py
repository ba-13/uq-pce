from uqpylab import sessions, display_general, display_util
import numpy as np
import matplotlib.pyplot as plt
from uq_pce.uqlab.base import (INPUT_DISTRIBUTION, init_uqlab, 
                               call_option_price, put_option_price, 
                               ModelTypeUQL, model_inputs, 
                               create_pce, eval_pce, create_lra)
from uq_pce.uqlab.sensitivity_utils import (lra_sobol, pce_sobol,
                                            corr_sens, src_sens,
                                            perturbation, cotter,
                                            morris, borgonovo,
                                            sobol_sens, ancova_sens)

Instance = 'https://uqcloud.ethz.ch'
Token = '54670830aaf970cdc8b7dd7e9015d308665f9096'

Session  = sessions.cloud(host=Instance,token=Token)
degree = 3

uq = init_uqlab(Session=Session,seed=42)
Model, Input = model_inputs(uq,ModelTypeUQL.CALL)
PCE, _ = create_pce(uq,130,'LHS','OLS',Model,degree)
LRA = create_lra(uq,1e4,Model,Input,degree)
N = 1e4

#corr_sens(uq,N,Input)
#src_sens(uq,N,Input)
#perturbation(uq,Input)
##cotter(uq,Input)
##morris(uq,0.5,N,N)
##borgonovo(uq,N,Input)
#sobol_sens(uq,N,degree)
pce_sobol(uq,degree,PCE)
##lra_sobol(uq,degree=degree)
##ancova_sens(uq,N)

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