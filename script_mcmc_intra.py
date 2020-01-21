import os
#import sys
import numpy as np
import pickle

from mxlMcmcIntra import estimate, predictB, predictW
from simstudy import rmse, tvd

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
N = 1000
T = 20
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

#Random parameter distributions
#0: normal
#1: log-normal
#2: S_B
xRnd_trans = np.array([0, 0])

zeta_inits = np.zeros((xRnd.shape[1],))
OmegaB_inits = 0.1 * np.eye(xRnd.shape[1])
OmegaW_inits = 0.1 * np.eye(xRnd.shape[1])

A = 1e3
nu = 2
diagCov = (False, False)

mcmc_nChain = 2
mcmc_iterBurn = 200000
mcmc_iterSample = 200000
mcmc_thin = 10
mcmc_iterMem = int(mcmc_iterSample / 10)
mcmc_disp = 1000
seed = 4711
simLogLik = False
simLogLikDrawsType = 'mlhs'
simDraws = 200  

rho = 0.1

modelName = 'd_mcmc_intra_' + filename
deleteDraws = False

results = estimate(
        mcmc_nChain, mcmc_iterBurn, mcmc_iterSample, mcmc_thin, mcmc_iterMem, mcmc_disp, 
        seed, simLogLik, simLogLikDrawsType, simDraws,
        rho,
        modelName, deleteDraws,
        A, nu, diagCov,
        zeta_inits, OmegaB_inits, OmegaW_inits,
        indID, obsID, altID, chosen,
        xRnd, xRnd_trans)

###
#Prediction: Between
###

nTakes = 1
nSim = 200

mcmc_thinPred = 2
mcmc_disp = 1000
deleteDraws = False

pPredB = predictB(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_thinPred, mcmc_disp, nTakes, nSim,
        seed,
        modelName, deleteDraws,
        indID_valB, obsID_valB, altID_valB, chosen_valB,
        xRnd_valB)

###
#Prediction: Within
###

nTakes = 10
nSim = 1000

mcmc_thinPred = 2
mcmc_disp = 1000
deleteDraws = True

pPredW = predictW(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_thinPred, mcmc_disp, nTakes, nSim,
        seed,
        modelName, deleteDraws,
        indID_valW, obsID_valW, altID_valW, chosen_valW,
        xRnd_valW)

###
#Evaluate results
###

res = np.zeros((6,))

res[0] = results['estimation_time']
res[4] = tvd(pPredB, pPredB_true)
res[5] = tvd(pPredW, pPredW_true)

###
#Parameter recovery
###

zetaMu = results['postMean_zeta']
res[1] = rmse(zetaMu, betaRndInd_true.mean(axis = 0))

tril_idx = np.tril_indices(nRnd)
SigmaB = results['postMean_OmegaB']
SigmaW = results['postMean_OmegaW']
res[2] = rmse(SigmaB[tril_idx], np.cov(betaRndInd_true, rowvar = False)[tril_idx])
res[3] = rmse(SigmaW[tril_idx], np.cov(betaRndObs_true, rowvar = False)[tril_idx])


###
#Save results
###

resList = [res, results]

filename = 'results/' + 'mcmc_intra_' + 'sim' + \
'_S' + str(S) + '_N' + str(N) + '_T' + str(T) + '_r' + str(r)
if os.path.exists(filename): os.remove(filename) 
outfile = open(filename, 'wb')
pickle.dump(resList, outfile)
outfile.close()