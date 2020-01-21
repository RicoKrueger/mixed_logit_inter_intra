import os
#import sys
import numpy as np
import pickle

from mxlMcmc import estimate, predictB, predictW
from simstudy import tvd

###
#Obtain task
###

task = int(os.getenv('TASK'))

filename = "taskplan"
infile = open(filename, 'rb')
taskplan = pickle.load(infile)
infile.close()

S = taskplan[task, 0]
N = taskplan[task, 1]
T = taskplan[task, 2]
r = taskplan[task, 3]

"""
S = 1
N = 250
T = 5
r = 0
"""

###
#Load data
###

filename = 'sim' + '_S' + str(S) + '_N' + str(N) + '_T' + str(T) + '_r' + str(r)
infile = open(filename, 'rb')
sim = pickle.load(infile)
infile.close()

locals().update(sim)

###
#Estimate MXL via MCMC
###

xFix = np.zeros((0,0))

#Fixed parameter distributions
#0: normal
#1: log-normal (to assure that fixed parameter is striclty negative or positive)
xFix_trans = np.array([0, 0])

#Random parameter distributions
#0: normal
#1: log-normal
#2: S_B
xRnd_trans = np.array([0, 0])

paramFix_inits = np.zeros((xFix.shape[1],))
zeta_inits = np.zeros((xRnd.shape[1],))
Omega_inits = 0.1 * np.eye(xRnd.shape[1])

A = 1e3
nu = 2
diagCov = False

mcmc_nChain = 2
mcmc_iterBurn = 50000
mcmc_iterSample = 50000
mcmc_thin = 5
mcmc_iterMem = mcmc_iterSample
mcmc_disp = 1000
seed = 4711
simDraws = 100  

rho = 0.1
rhoF = 0.01

modelName = 'd_mcmc_inter_' + filename
deleteDraws = False

results = estimate(
        mcmc_nChain, mcmc_iterBurn, mcmc_iterSample, mcmc_thin, mcmc_iterMem, mcmc_disp, 
        seed, simDraws,
        rhoF, rho,
        modelName, deleteDraws,
        A, nu, diagCov,
        paramFix_inits, zeta_inits, Omega_inits,
        indID, obsID, altID, chosen,
        xFix, xRnd,
        xFix_trans, xRnd_trans)


###
#Prediction: Between
###

nTakes = 10
nSim = 1000

mcmc_thinPred = 1
mcmc_disp = 1000
deleteDraws = False

xFix_valB = np.zeros((0,0))

pPredB = predictB(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_thinPred, mcmc_disp, nTakes, nSim,
        seed,
        modelName, deleteDraws,
        indID_valB, obsID_valB, chosen_valB,
        xFix_valB, xRnd_valB)

###
#Prediction: Within
###

mcmc_thinPred = 1
mcmc_disp = 1000
deleteDraws = True

xFix_valW = np.zeros((0,0))

pPredW = predictW(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_thinPred, mcmc_disp,
        seed,
        modelName, deleteDraws,
        indID_valW, obsID_valW, chosen_valW,
        xFix_valW, xRnd_valW)

###
#Evaluate results
###

res = np.zeros((6,))

res[0] = results['estimation_time']
res[4] = tvd(pPredB, pPredB_true)
res[5] = tvd(pPredW, pPredW_true)

###
#Save results
###

resList = [res, results]

filename = 'results/' + 'mcmc_inter_' + 'sim' + \
'_S' + str(S) + '_N' + str(N) + '_T' + str(T) + '_r' + str(r)
if os.path.exists(filename): os.remove(filename) 
outfile = open(filename, 'wb')
pickle.dump(resList, outfile)
outfile.close()