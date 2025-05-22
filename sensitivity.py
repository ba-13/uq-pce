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
N = 1e4

## UN-COMMENT DESIRED METHOD TO RUN SENSITIVITY ANALYSIS ##
# For Cotter and Borgonovo: 
# Non-Zero Standard Deviation for Variable 't' Recommended

#corr_sens(uq,N,Input)
#src_sens(uq,N,Input)
#perturbation(uq,Input)
#cotter(uq) ##only works with non-zero 't' input stDev!
#morris(uq,N,N)
#borgonovo(uq,N) ##only works with non-zero 't' input stDev!
#sobol_sens(uq,N,degree,'LHS') #sampling options: 'LHS', 'MC'
pce_sobol(uq,degree,PCE)
#ancova_sens(uq,200)

Session.quit()

## low-rank approximation sobol indices (not used)
#LRA = create_lra(uq,1e4,Model,Input,degree)
#lra_sobol(uq,degree=degree)