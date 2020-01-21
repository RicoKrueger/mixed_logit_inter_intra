import os
#import sys
import numpy as np
import pickle

from mxlVbIntra import estimate, inits, predictB, predictW
from simstudy import rmse,tvd

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
#Estimate MXL via VB
###

mu0Rnd = np.zeros((xRnd.shape[1],))
Sigma0RndInv = 1e-6 * np.eye(xRnd.shape[1]) 
nu = 2; A = 1e3;

(paramRndMuB_inits, paramRndSiB_inits,
 paramRndMuW_inits, paramRndSiW_inits,
 zetaMu_inits, zetaSi_inits,
 cK, rK,
 omegaB, psiB_inits, dK_B_inits,
 omegaW, psiW_inits, dK_W_inits) = inits(indID, obsID, xRnd, nu, A)

svb = False
vb = True

vb_iter = 1000
vb_tol = 0.005

local_iter = vb_iter
local_tol = vb_tol

svb_eta = 1
svb_kappa = 250

K = (4, int(N / 125))
N_K = 100
T_K = T

drawsType = 'mlhs'
nDrawsB = 100
nDrawsW_m = 1

intra = True

modelName = 'test'
seed = 4711

results = estimate(
        svb, vb,
        vb_iter, vb_tol, modelName, seed,
        local_iter, local_tol,
        svb_eta, svb_kappa,
        K, N_K, T_K,
        drawsType, nDrawsB, nDrawsW_m,
        paramRndMuB_inits, paramRndSiB_inits,
        paramRndMuW_inits, paramRndSiW_inits,
        zetaMu_inits, zetaSi_inits,
        cK, rK,
        omegaB, psiB_inits, dK_B_inits,
        omegaW, psiW_inits, dK_W_inits,
        mu0Rnd, Sigma0RndInv, nu, A,
        indID, obsID, chosen,
        xRnd, intra) 

###
#Prediction: Between
###

zetaMu = results['zetaMu'] 
zetaSi = results['zetaSi'] 
psiB = results['psiB']
psiW = results['psiW']

nIter = 1000
nTakes = 1
nSim = 200

pPredB = predictB(
        nIter, nTakes, nSim, seed,
        zetaMu, zetaSi, psiB, omegaB, psiW, omegaW,
        indID_valB, obsID_valB, altID_valB, chosen_valB,
        xRnd_valB)

###
#Prediction: Within
###

nValW = np.unique(indID_valW).shape[0]

paramRndMuB = np.array(results['paramRndMuB'][:nValW,:])
paramRndSiB = np.array(results['paramRndSiB'][:nValW,:])
psiW = results['psiW']

nIter = 1000
nTakes = 10
nSim = 1000

pPredW = predictW(
        nIter, nTakes, nSim, seed,
        paramRndMuB, paramRndSiB, psiW, omegaW,
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

zetaMu = results['zetaMu']
res[1] = rmse(zetaMu, betaRndInd_true.mean(axis = 0))

tril_idx = np.tril_indices(nRnd)
SigmaB = psiB / omegaB
SigmaW = psiW / omegaW
res[2] = rmse(SigmaB[tril_idx], np.cov(betaRndInd_true, rowvar = False)[tril_idx])
res[3] = rmse(SigmaW[tril_idx], np.cov(betaRndObs_true, rowvar = False)[tril_idx])

###
#Save results
###

resList = [res, results]

filename = 'results/' + 'vb_intra_' + 'sim' + \
'_S' + str(S) + '_N' + str(N) + '_T' + str(T) + '_r' + str(r)
if os.path.exists(filename): os.remove(filename) 
outfile = open(filename, 'wb')
pickle.dump(resList, outfile)
outfile.close()