from joblib import Parallel, delayed
import os
import sys
import time
import pandas as pd
import numpy as np
from scipy.stats import invwishart
import scipy.sparse
from math import floor
import h5py

from mxl import corrcov, prepareData, transRnd, pPredMxl
from mxlMcmc import next_zeta, next_iwDiagA, next_Omega
from qmc import makeNormalDraws

###
#Probabilities
###
    
def mvnlpdfIntra(x, mu, Sigma, obsPerInd):
    xS = (x - np.repeat(mu, obsPerInd,  axis = 0)).T
    f = -0.5 * (xS * np.linalg.solve(Sigma, xS)).sum(axis = 0)
    return f
    
def probMxl(
        paramRndW,
        xRnd, xRnd_transBool, xRnd_trans,
        rowsPerObs, map_avail_to_obs):
    
    if xRnd_transBool: paramRndW = transRnd(paramRndW, xRnd_trans)
    paramRndWPerRow = np.repeat(paramRndW, rowsPerObs, axis = 0)
    v = np.sum(xRnd * paramRndWPerRow, axis = 1)
            
    ev = np.exp(v)
    ev[ev > 1e+200] = 1e+200
    ev[ev < 1e-300] = 1e-300 
    nev = map_avail_to_obs.T @ ev + 1
    pChosen = 1 / nev
    lPChosen = np.log(pChosen)
    return pChosen, lPChosen
                
###
#MCMC
###
    

def next_OmegaW(paramRndW, paramRndB, nu, iwDiagA_W, diagCov, nRnd, nObs, obsPerInd):
    betaS = paramRndW - np.repeat(paramRndB, obsPerInd, axis = 0)
    OmegaW = np.array(invwishart.rvs(nu + nObs + nRnd - 1, 2 * nu * np.diag(iwDiagA_W) + betaS.T @ betaS)).reshape((nRnd, nRnd))
    if diagCov: OmegaW = np.diag(np.diag(OmegaW))
    return OmegaW

def next_paramRndB(
        paramRndW, zeta, OmegaB, OmegaW, 
        nInd, nRnd, obsPerInd, map_obs_to_ind):
    SigmaInv = np.linalg.inv(OmegaB) + obsPerInd[0] * np.linalg.inv(OmegaW)
    mu_n_b = np.linalg.solve(OmegaB, zeta).reshape((nRnd,1)) + \
    np.linalg.solve(OmegaW, (paramRndW.T @ map_obs_to_ind))
    mu_n = np.linalg.solve(SigmaInv, mu_n_b)
    paramRndB = (mu_n + np.linalg.solve(np.linalg.cholesky(SigmaInv).T, 
                                        np.random.randn(nRnd, nInd))).T
    return paramRndB

def next_paramRndW(
        paramRndW, paramRndB, OmegaW,
        lPChosen,
        xRnd, xRnd_transBool, xRnd_trans, nRnd,
        nObs, obsPerInd, rowsPerObs, map_avail_to_obs,
        rho):
    lPhi = mvnlpdfIntra(paramRndW, paramRndB, OmegaW, obsPerInd)
    paramRndW_star = paramRndW + np.sqrt(rho) * (np.linalg.cholesky(OmegaW) @ np.random.randn(nRnd, nObs)).T    
    _, lPChosen_star = probMxl(
        paramRndW_star,
        xRnd, xRnd_transBool, xRnd_trans,
        rowsPerObs, map_avail_to_obs)
    lPhi_star = mvnlpdfIntra(paramRndW_star, paramRndB, OmegaW, obsPerInd)

    r = np.exp(lPChosen_star + lPhi_star - lPChosen - lPhi)
    idxAccept = np.random.rand(nObs,) <= r

    paramRndW[idxAccept, :] = np.array(paramRndW_star[idxAccept, :])
    lPChosen[idxAccept] = np.array(lPChosen_star[idxAccept])

    acceptRate = np.mean(idxAccept)
    rho = rho - 0.001 * (acceptRate < 0.3) + 0.001 * (acceptRate > 0.3)
    return paramRndW, lPChosen, rho, acceptRate

def mcmcChain(
        chainID, seed,
        mcmc_iter, mcmc_iterBurn, mcmc_iterSampleThin, mcmc_iterMemThin, mcmc_thin, mcmc_disp,
        rho,
        modelName,
        zeta, OmegaB, OmegaW, invASq, nu, diagCov,
        xRnd, xRnd_transBool, xRnd_trans, nRnd, 
        nInd, nObs, obsPerInd, rowsPerObs, map_obs_to_ind, map_avail_to_obs):   
    
    np.random.seed(seed + chainID)
    
    ###
    #Precomputations
    ###
    
    iwDiagA_B = np.random.gamma(1 / 2, 1 / invASq)
    iwDiagA_W = np.random.gamma(1 / 2, 1 / invASq)
    paramRndB = zeta + (np.linalg.cholesky(OmegaB) @ np.random.randn(nRnd, nInd)).T
    paramRndW = np.repeat(paramRndB, obsPerInd, axis = 0) + (np.linalg.cholesky(OmegaW) @ np.random.randn(nRnd, nObs)).T

    _, lPChosen = probMxl(
            paramRndW,
            xRnd, xRnd_transBool, xRnd_trans,
            rowsPerObs, map_avail_to_obs)   
    
    ###
    #Storage
    ###
    
    fileName = modelName + '_draws_chain' + str(chainID + 1) + '.hdf5'
    if os.path.exists(fileName):
        os.remove(fileName) 
    file = h5py.File(fileName, "a")
    
    paramRndB_store = file.create_dataset('paramRndB_store', (mcmc_iterSampleThin, nInd, nRnd), dtype='float64')
    zeta_store = file.create_dataset('zeta_store', (mcmc_iterSampleThin, nRnd), dtype='float64')
    OmegaB_store = file.create_dataset('OmegaB_store', (mcmc_iterSampleThin, nRnd, nRnd), dtype='float64')
    CorrB_store = file.create_dataset('CorrB_store', (mcmc_iterSampleThin, nRnd, nRnd), dtype='float64')
    sdB_store = file.create_dataset('sdB_store', (mcmc_iterSampleThin, nRnd), dtype='float64')
    OmegaW_store = file.create_dataset('OmegaW_store', (mcmc_iterSampleThin, nRnd, nRnd), dtype='float64')
    CorrW_store = file.create_dataset('CorrW_store', (mcmc_iterSampleThin, nRnd, nRnd), dtype='float64')
    sdW_store = file.create_dataset('sdW_store', (mcmc_iterSampleThin, nRnd), dtype='float64')
    
    paramRndB_store_tmp = np.zeros((mcmc_iterMemThin, nInd, nRnd))
    zeta_store_tmp = np.zeros((mcmc_iterMemThin, nRnd))
    OmegaB_store_tmp = np.zeros((mcmc_iterMemThin, nRnd, nRnd))
    CorrB_store_tmp = np.zeros((mcmc_iterMemThin, nRnd, nRnd))
    sdB_store_tmp = np.zeros((mcmc_iterMemThin, nRnd))
    OmegaW_store_tmp = np.zeros((mcmc_iterMemThin, nRnd, nRnd))
    CorrW_store_tmp = np.zeros((mcmc_iterMemThin, nRnd, nRnd))
    sdW_store_tmp = np.zeros((mcmc_iterMemThin, nRnd))
    
    ###
    #Sample
    ###
    
    j = -1
    ll = 0
    acceptRateAvg = 0
    sampleState = 'burn in'
    for i in np.arange(mcmc_iter):
        
        iwDiagA_B = next_iwDiagA(OmegaB, nu, invASq, nRnd)
        OmegaB = next_Omega(paramRndB, zeta, nu, iwDiagA_B, 
                            diagCov[0], nRnd, nInd) 
        
        iwDiagA_W = next_iwDiagA(OmegaW, nu, invASq, nRnd)
        OmegaW = next_OmegaW(paramRndW, paramRndB, nu, iwDiagA_W, 
                             diagCov[1], nRnd, nObs, obsPerInd)     
        
        zeta = next_zeta(paramRndB, OmegaB, nRnd, nInd)
        
        paramRndB = next_paramRndB(
                paramRndW, zeta, OmegaB, OmegaW, 
                nInd, nRnd, obsPerInd, map_obs_to_ind)
        paramRndW, lPChosen, rho, acceptRate = next_paramRndW(
                paramRndW, paramRndB, OmegaW,
                lPChosen,
                xRnd, xRnd_transBool, xRnd_trans, nRnd,
                nObs, obsPerInd, rowsPerObs, map_avail_to_obs,
                rho)
        
        acceptRateAvg += acceptRate
        
        if ((i + 1) % mcmc_disp) == 0:
            if (i + 1) > mcmc_iterBurn: sampleState = 'sampling'
            acceptRateAvg /= mcmc_disp
            print('Chain ' + str(chainID + 1) + '; iteration: ' + str(i + 1) + ' (' + sampleState + '); Avg. accept rate: ' + str(acceptRateAvg))
            acceptRateAvg = 0
            sys.stdout.flush()
            
        if (i + 1) > mcmc_iterBurn:   
            if ((i + 1) % mcmc_thin) == 0:
                j+=1
            
                paramRndB_store_tmp[j,:,:] = paramRndB
                zeta_store_tmp[j,:] = zeta
                OmegaB_store_tmp[j,:,:] = OmegaB
                CorrB_store_tmp[j,:,:], sdB_store_tmp[j,:] = corrcov(OmegaB)
                OmegaW_store_tmp[j,:,:] = OmegaW
                CorrW_store_tmp[j,:,:], sdW_store_tmp[j,:] = corrcov(OmegaW)
                    
            if (j + 1) == mcmc_iterMemThin:
                l = ll; ll += mcmc_iterMemThin; sl = slice(l, ll)
                
                print('Storing chain ' + str(chainID + 1))
                sys.stdout.flush()
                
                paramRndB_store[sl,:,:] = paramRndB_store_tmp
                zeta_store[sl,:] = zeta_store_tmp
                OmegaB_store[sl,:,:] = OmegaB_store_tmp
                CorrB_store[sl,:,:] = CorrB_store_tmp
                sdB_store[sl,:,] = sdB_store_tmp
                OmegaW_store[sl,:,:] = OmegaW_store_tmp
                CorrW_store[sl,:,:] = CorrW_store_tmp
                sdW_store[sl,:,] = sdW_store_tmp
                
                j = -1 

###
#Posterior analysis
###  

def postAna(paramName, nParam, nParam2, mcmc_nChain, mcmc_iterSampleThin, modelName):
    colHeaders = ['mean', 'std. dev.', '2.5%', '97.5%', 'Rhat']
    q = np.array([0.025, 0.975])
    nSplit = 2
    
    postDraws = np.zeros((mcmc_nChain, mcmc_iterSampleThin, nParam, nParam2))
    for c in range(mcmc_nChain):
        file = h5py.File(modelName + '_draws_chain' + str(c + 1) + '.hdf5', 'r')
        postDraws[c,:,:,:] = np.array(file[paramName + '_store']).reshape((mcmc_iterSampleThin, nParam, nParam2))
        
    tabPostAna = np.zeros((nParam * nParam2, len(colHeaders)))
    postMean = np.mean(postDraws, axis = (0,1))
    tabPostAna[:, 0] = np.array(postMean).reshape((nParam * nParam2,))
    tabPostAna[:, 1] = np.array(np.std(postDraws, axis = (0,1))).reshape((nParam * nParam2,))
    tabPostAna[:, 2] = np.array(np.quantile(postDraws, q[0], axis = (0,1))).reshape((nParam * nParam2,))
    tabPostAna[:, 3] = np.array(np.quantile(postDraws, q[1], axis = (0,1))).reshape((nParam * nParam2,))
    
    m = int(mcmc_nChain * nSplit)
    n = int(mcmc_iterSampleThin / nSplit)
    postDrawsSplit = np.zeros((m, n, nParam, nParam2))
    postDrawsSplit[:mcmc_nChain,:,:,:] = postDraws[:,:n,:,:]
    postDrawsSplit[mcmc_nChain:,:,:,:] = postDraws[:,n:,:,:]
    muChain = np.mean(postDrawsSplit, axis = 1).reshape((m, 1, nParam, nParam2))
    mu = np.mean(muChain, axis = 0).reshape((1, 1, nParam, nParam2))
    B = (n / (m - 1)) * np.sum((muChain - mu)**2, axis=(0,1))
    ssq = (1 / (n - 1)) * np.sum((postDrawsSplit - muChain)**2, axis = 1)
    W = np.mean(ssq, axis = 0)
    varPlus = ((n - 1) / n) * W + B / n
    Rhat = np.empty((nParam, nParam2)) * np.nan
    W_idx = W > 0
    Rhat[W_idx] = np.sqrt(varPlus[W_idx] / W[W_idx])
    tabPostAna[:, 4] = np.array(Rhat).reshape((nParam * nParam2,))
    
    if paramName not in ["OmegaB", "CorrB", "paramRndB",
                         "OmegaW", "CorrW"]:
        postMean = np.ndarray.flatten(postMean)
        
    pdTabPostAna = pd.DataFrame(tabPostAna, columns = colHeaders) 
    return postMean, pdTabPostAna             

###
#Estimate
###
    
def estimate(
        mcmc_nChain, mcmc_iterBurn, mcmc_iterSample, mcmc_thin, mcmc_iterMem, mcmc_disp, 
        seed, simLogLik, simLogLikDrawsType, simDraws,
        rho,
        modelName, deleteDraws,
        A, nu, diagCov,
        zeta_inits, OmegaB_inits, OmegaW_inits,
        indID, obsID, altID, chosen,
        xRnd, xRnd_trans):
    ###
    #Prepare data
    ###
    
    nRnd = xRnd.shape[1]

    xRnd_transBool = np.sum(xRnd_trans) > 0  
    
    xList = [xRnd]
    (xList,
     nInd, nObs, nRow,
     chosenIdx, nonChosenIdx,
     rowsPerInd, rowsPerObs,
     map_obs_to_ind, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen)
    xRnd = xList[0]
    
    obsPerInd = np.ones((nObs,), dtype = 'int64') @ map_obs_to_ind
    
    assert np.unique(obsPerInd).shape[0] == 1, \
    "Number of choice tasks must be the same for all decision makers!"
    
    ### 
    #Posterior sampling
    ###
    
    mcmc_iter = mcmc_iterBurn + mcmc_iterSample
    mcmc_iterSampleThin = floor(mcmc_iterSample / mcmc_thin)
    mcmc_iterMemThin = floor(mcmc_iterMem / mcmc_thin)

    A = A * np.ones((nRnd,))
    invASq = A ** (-2)
    
    zeta = zeta_inits
    OmegaB = OmegaB_inits
    OmegaW = OmegaW_inits
    
    tic = time.time()
	
    Parallel(n_jobs = mcmc_nChain)(delayed(mcmcChain)(
                c, seed,
                mcmc_iter, mcmc_iterBurn, mcmc_iterSampleThin, mcmc_iterMemThin, mcmc_thin, mcmc_disp,
                rho,
                modelName,
                zeta, OmegaB, OmegaW, invASq, nu, diagCov,
                xRnd, xRnd_transBool, xRnd_trans, nRnd, 
                nInd, nObs, obsPerInd, rowsPerObs, map_obs_to_ind, map_avail_to_obs) 
    for c in range(mcmc_nChain))

    toc = time.time() - tic
    
    print(' ')
    print('Computation time [s]: ' + str(toc))
        
    ###
    #Posterior analysis
    ###
 
    postMean_zeta, pdTabPostAna_zeta = postAna('zeta', nRnd, 1, mcmc_nChain, mcmc_iterSampleThin, modelName)
    print(' ')
    print('Random parameters (means):')    
    print(pdTabPostAna_zeta)
    
    postMean_sdB, pdTabPostAna_sdB = postAna('sdB', nRnd, 1, mcmc_nChain, mcmc_iterSampleThin, modelName)
    print(' ')
    print('Random parameters (standard deviations; between):')    
    print(pdTabPostAna_sdB)
    
    postMean_OmegaB, pdTabPostAna_OmegaB = postAna('OmegaB', nRnd, nRnd, mcmc_nChain, mcmc_iterSampleThin, modelName)
    print(' ')
    print('Random parameters (covariance matrix; between):')    
    print(pdTabPostAna_OmegaB)
    
    postMean_CorrB, pdTabPostAna_CorrB = postAna('CorrB', nRnd, nRnd, mcmc_nChain, mcmc_iterSampleThin, modelName)
    print(' ')
    print('Random parameters (correlation matrix; between):')    
    print(pdTabPostAna_CorrB)
    
    postMean_sdW, pdTabPostAna_sdW = postAna('sdW', nRnd, 1, mcmc_nChain, mcmc_iterSampleThin, modelName)
    print(' ')
    print('Random parameters (standard deviations; within):')    
    print(pdTabPostAna_sdW)
    
    postMean_OmegaW, pdTabPostAna_OmegaW = postAna('OmegaW', nRnd, nRnd, mcmc_nChain, mcmc_iterSampleThin, modelName)
    print(' ')
    print('Random parameters (covariance matrix; within):')    
    print(pdTabPostAna_OmegaW)
    
    postMean_CorrW, pdTabPostAna_CorrW = postAna('CorrW', nRnd, nRnd, mcmc_nChain, mcmc_iterSampleThin, modelName)
    print(' ')
    print('Random parameters (correlation matrix; within):')    
    print(pdTabPostAna_CorrW)   
    
    postMean_paramRndB, pdTabPostAna_paramRndB = postAna('paramRndB', nInd, nRnd, mcmc_nChain, mcmc_iterSampleThin, modelName)
    
    ###
    #Simulate log-likelihood at posterior means
    ###
    
    if simLogLik:
        print(' ')
        np.random.seed(seed)
        
        postMean_chOmegaB = np.linalg.cholesky(postMean_OmegaB)     
        postMean_chOmegaW = np.linalg.cholesky(postMean_OmegaW)
    
        _, drawsB = makeNormalDraws(simDraws, nRnd, simLogLikDrawsType, nInd)
        drawsB = np.array(drawsB).reshape((simDraws, nInd, nRnd))
        _, drawsW = makeNormalDraws(simDraws, nRnd, simLogLikDrawsType, nObs)  
        drawsW = np.array(drawsW).reshape((simDraws, nObs, nRnd))
           
        pIndSim = 0         
        for i in np.arange(simDraws):
            paramRndB = postMean_zeta + (postMean_chOmegaB @ drawsB[i,:,:].T).T
            paramRndB_perObs = np.repeat(paramRndB, obsPerInd, axis = 0)
            
            pObsSim = 0
            for j in np.arange(simDraws):
                paramRndW = paramRndB_perObs + (postMean_chOmegaW @ drawsW[j,:,:].T).T
                pChosen, _ = probMxl(
                        paramRndW,
                        xRnd, xRnd_transBool, xRnd_trans,
                        rowsPerObs, map_avail_to_obs)
                pObsSim += pChosen
            pObsSim /= simDraws
            pIndSim += np.exp(np.log(pObsSim) @ map_obs_to_ind)
            
            if ((i + 1) % 50) == 0:
                print('Log-likelihood simulation (inter-ind. draws): ' + str(i + 1))
            sys.stdout.flush()
            
        pIndSim /= simDraws
        
        logLik = np.sum(np.log(pIndSim))
        print(' ')
        print('Log-likelihood (simulated at posterior means): ' + str(logLik))
    else:
        logLik = None
    
    ###
    #Delete draws
    ###
    
    if deleteDraws:
        for c in range(mcmc_nChain):
            os.remove(modelName + '_draws_chain' + str(c + 1) + '.hdf5') 
        
    ###
    #Save results
    ###
    
    results = {'modelName': modelName, 'seed': seed,
               'estimation_time': toc,
               'logLik': logLik,
               'postMean_paramRndB': postMean_paramRndB, 'pdTabPostAna_paramRndB': pdTabPostAna_paramRndB,
               'postMean_zeta': postMean_zeta, 'pdTabPostAna_zeta': pdTabPostAna_zeta, 
               'postMean_sdB': postMean_sdB, 'pdTabPostAna_sdB': pdTabPostAna_sdB, 
               'postMean_OmegaB': postMean_OmegaB, 'pdTabPostAna_OmegaB': pdTabPostAna_OmegaB, 
               'postMean_CorrB': postMean_CorrB, 'pdTabPostAna_CorrB': pdTabPostAna_CorrB,
               'postMean_sdW': postMean_sdW, 'pdTabPostAna_sdW': pdTabPostAna_sdW, 
               'postMean_OmegaW': postMean_OmegaW, 'pdTabPostAna_OmegaW': pdTabPostAna_OmegaW, 
               'postMean_CorrW': postMean_CorrW, 'pdTabPostAna_CorrW': pdTabPostAna_CorrW,
               }
    
    return results

###
#Prediction: Between
###
    
def mcmcChainPredB(
        chainID, seed,
        mcmc_iterSampleThin, mcmc_iterSampleThinPred, mcmc_disp, nTakes, nSim,
        modelName,
        sim_xRnd, nRnd, 
        nInd, nObs, nRow,
        rowsPerInd, sim_rowsPerObs, sim_map_avail_to_obs, chosenIdx, nonChosenIdx):   
    
    np.random.seed(seed + chainID)
    
    ###
    #Retrieve draws
    ###
    
    fileName = modelName + '_draws_chain' + str(chainID + 1) + '.hdf5'
    file = h5py.File(fileName, "r")
    
    zeta_store = np.array(file['zeta_store'])
    OmegaB_store = np.array(file['OmegaB_store'])
    OmegaW_store = np.array(file['OmegaW_store'])
    
    ###
    #Simulate
    ###

    pPred = np.zeros((nRow + nObs,))
    vFix = 0 
    
    sampleIdx = np.sort(np.random.choice(np.arange(mcmc_iterSampleThin), 
                                         size = mcmc_iterSampleThinPred, 
                                         replace = False))
    
    ii = -1
    for i in sampleIdx:
        ii += 1
        
        zeta_tmp = zeta_store[i,:]
        chB_tmp = np.linalg.cholesky(OmegaB_store[i,:,:])
        chW_tmp = np.linalg.cholesky(OmegaW_store[i,:,:])
        
        pPred_iter = np.zeros((nRow + nObs,))
        
        for j in np.arange(nSim * nTakes):
            
            betaRndInd = zeta_tmp.reshape((1,nRnd)) + (chB_tmp @ np.random.randn(nRnd, nInd)).T
            betaRndInd_perRow = np.tile(np.repeat(betaRndInd, rowsPerInd, axis = 0), (nSim, 1))
                
            for k in np.arange(nTakes):
                betaRndObs = (chW_tmp @ np.random.randn(nRnd, nObs * nSim)).T
                betaRnd = betaRndInd_perRow + np.repeat(betaRndObs, sim_rowsPerObs, axis = 0)
                vRnd = np.sum(sim_xRnd * betaRnd, axis = 1)
                    
                pPred_take = pPredMxl(vFix, vRnd, sim_map_avail_to_obs, nSim, chosenIdx, nonChosenIdx)
                pPred_iter += pPred_take 
        
        pPred += pPred_iter / (nSim * nTakes**2)
        
        if ((ii + 1) % mcmc_disp) == 0:
            print('Chain ' + str(chainID + 1) + '; iteration: ' + str(ii + 1) + ' (predictive simulation)')
            sys.stdout.flush()
            
    pPred /= mcmc_iterSampleThinPred
    return pPred
    
def predictB(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_thinPred, mcmc_disp, nTakes, nSim,
        seed,
        modelName, deleteDraws,
        indID, obsID, altID, chosen,
        xRnd):
    ###
    #Prepare data
    ###
    
    nRnd = xRnd.shape[1]
    
    xList = [xRnd]
    (xList,
     nInd, nObs, nRow,
     chosenIdx, nonChosenIdx,
     rowsPerInd, rowsPerObs,
     map_obs_to_ind, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen)
    xRnd = xList[0]
    
    sim_xRnd = np.tile(xRnd, (nSim, 1))
    sim_rowsPerObs = np.tile(rowsPerObs, (nSim,))
    sim_map_avail_to_obs = scipy.sparse.kron(scipy.sparse.eye(nSim), map_avail_to_obs)
    
    ### 
    #Predictive simulation
    ###
    
    mcmc_iterSampleThin = floor(mcmc_iterSample / mcmc_thin)
    mcmc_iterSampleThinPred = floor(mcmc_iterSampleThin / mcmc_thinPred)
    
    pPred = Parallel(n_jobs = mcmc_nChain)(delayed(mcmcChainPredB)(
            c, seed,
            mcmc_iterSampleThin, mcmc_iterSampleThinPred, mcmc_disp, nTakes, nSim,
            modelName,
            sim_xRnd, nRnd, 
            nInd, nObs, nRow,
            rowsPerInd, sim_rowsPerObs, sim_map_avail_to_obs, chosenIdx, nonChosenIdx) 
    for c in range(mcmc_nChain))
    pPred = np.array(pPred).mean(axis = 0)
    
    ###
    #Delete draws
    ###
    
    if deleteDraws:
        for c in range(mcmc_nChain):
            os.remove(modelName + '_draws_chain' + str(c + 1) + '.hdf5') 

    return pPred

###
#Prediction: Within
###
    
def mcmcChainPredW(
        chainID, seed,
        mcmc_iterSampleThin, mcmc_iterSampleThinPred, mcmc_disp, nTakes, nSim,
        modelName,
        sim_xRnd, nRnd, 
        nInd, nObs, nRow,
        obsPerInd, sim_rowsPerObs, sim_map_avail_to_obs, chosenIdx, nonChosenIdx):   
    
    np.random.seed(seed + chainID)
    
    ###
    #Retrieve draws
    ###
    
    fileName = modelName + '_draws_chain' + str(chainID + 1) + '.hdf5'
    file = h5py.File(fileName, "r")
    
    paramRndB_store = np.array(file['paramRndB_store'][:,:nInd,:])
    OmegaW_store = np.array(file['OmegaW_store'])
    
    ###
    #Simulate
    ###

    pPred = np.zeros((nRow + nObs,))
    vFix = 0 
    
    sampleIdx = np.sort(np.random.choice(np.arange(mcmc_iterSampleThin), 
                                         size = mcmc_iterSampleThinPred, 
                                         replace = False))
    
    ii = -1
    for i in sampleIdx:
        ii += 1
        
        paramRndB_tmp = np.tile(np.repeat(paramRndB_store[i,:,:], obsPerInd, axis = 0), (nSim, 1))  
        chW_tmp = np.linalg.cholesky(OmegaW_store[i,:,:])
        
        pPred_iter = np.zeros((nRow + nObs,))
        
        for t in np.arange(nTakes):
            paramRnd = paramRndB_tmp + (chW_tmp @ np.random.randn(nRnd, nObs * nSim)).T
            paramRndPerRow = np.repeat(paramRnd, sim_rowsPerObs, axis = 0)
            vRnd = np.sum(sim_xRnd * paramRndPerRow, axis = 1)
            
            pPred_take = pPredMxl(vFix, vRnd, sim_map_avail_to_obs, nSim, chosenIdx, nonChosenIdx)
            pPred_iter += pPred_take
            
        pPred += pPred_iter / nTakes
        
        if ((ii + 1) % mcmc_disp) == 0:
            print('Chain ' + str(chainID + 1) + '; iteration: ' + str(ii + 1) + ' (predictive simulation)')
            sys.stdout.flush()
            
    pPred /= mcmc_iterSampleThinPred
    return pPred
    
def predictW(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_thinPred, mcmc_disp, nTakes, nSim,
        seed,
        modelName, deleteDraws,
        indID, obsID, altID, chosen,
        xRnd):
    ###
    #Prepare data
    ###
    
    nRnd = xRnd.shape[1]
    
    xList = [xRnd]
    (xList,
     nInd, nObs, nRow,
     chosenIdx, nonChosenIdx,
     rowsPerInd, rowsPerObs,
     map_obs_to_ind, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen)
    xRnd = xList[0]
    
    obsPerInd = np.ones((nObs,), dtype = 'int64') @ map_obs_to_ind
    sim_xRnd = np.tile(xRnd, (nSim, 1))
    sim_rowsPerObs = np.tile(rowsPerObs, (nSim,))
    sim_map_avail_to_obs = scipy.sparse.kron(scipy.sparse.eye(nSim), map_avail_to_obs)
    
    ### 
    #Predictive simulation
    ###
    
    mcmc_iterSampleThin = floor(mcmc_iterSample / mcmc_thin)
    mcmc_iterSampleThinPred = floor(mcmc_iterSampleThin / mcmc_thinPred)
    
    pPred = Parallel(n_jobs = mcmc_nChain)(delayed(mcmcChainPredW)(
            c, seed,
            mcmc_iterSampleThin, mcmc_iterSampleThinPred, mcmc_disp, nTakes, nSim,
            modelName,
            sim_xRnd, nRnd, 
            nInd, nObs, nRow,
            obsPerInd, sim_rowsPerObs, sim_map_avail_to_obs, chosenIdx, nonChosenIdx) 
    for c in range(mcmc_nChain))
    pPred = np.array(pPred).mean(axis = 0)
    
    ###
    #Delete draws
    ###
    
    if deleteDraws:
        for c in range(mcmc_nChain):
            os.remove(modelName + '_draws_chain' + str(c + 1) + '.hdf5') 

    return pPred

###
#If main: test
###
    
if __name__ == "__main__":
    
    np.random.seed(4711)
    
    ###
    #Generate data
    ###
    
    N = 500
    T = 10
    NT = N * T
    J = 5
    NTJ = NT * J
    
    K = 2
    
    true_zeta = np.array([-0.8, 0.8, 1.0, -0.8, 1.5])
    """
    true_Omega = np.array([[1.0, 0.4, 0.4, 0.4, 0.4],
                           [0.4, 1.0, 0.4, 0.4, 0.4],
                           [0.4, 0.4, 1.0, 0.4, 0.4],
                           [0.4, 0.4, 0.4, 1.0, 0.4],
                           [0.4, 0.4, 0.4, 0.4, 1.0]])
    """
    true_OmegaW = np.array([[1.0, 0.2, 0.2, 0.2, 0.2],
                            [0.2, 1.0, 0.2, 0.2, 0.2],
                            [0.2, 0.2, 1.0, 0.2, 0.2],
                            [0.2, 0.2, 0.2, 1.0, 0.2],
                            [0.2, 0.2, 0.2, 0.2, 1.0]])

    true_Omega = np.array([[1.0, 0.8, 0.8, 0.8, 0.8],
                           [0.8, 1.0, 0.8, 0.8, 0.8],
                           [0.8, 0.8, 1.0, 0.8, 0.8],
                           [0.8, 0.8, 0.8, 1.0, 0.8],
                           [0.8, 0.8, 0.8, 0.8, 1.0]])

    pB, pW = 3/3, 3/3
    
    xRnd = np.random.rand(NTJ, K)

    betaInd = true_zeta[:K] + (np.linalg.cholesky(pB * true_Omega[:K,:K]) @ np.random.randn(K, N)).T
    betaObs = np.repeat(betaInd, T, axis = 0) + (np.linalg.cholesky(pW * true_OmegaW[:K,:K]) @ np.random.randn(K, N * T)).T    
    betaRow = np.repeat(betaObs, J, axis = 0)
    
    eps = -np.log(-np.log(np.random.rand(NTJ,)))
    
    vDet = np.sum(xRnd * betaRow, axis = 1)
    v = vDet + eps
    
    vDetMax = np.zeros((NT,))
    vMax = np.zeros((NT,))
    
    chosen = np.zeros((NTJ,), dtype = 'int64')
    
    for t in np.arange(NT):
        l = t * J; u = (t + 1) * J
        altMaxDet = np.argmax(vDet[l:u])
        altMax = np.argmax(v[l:u])
        vDetMax[t] = altMaxDet
        vMax[t] = altMax
        chosen[l + altMax] = 1
        
    error = np.sum(vMax == vDetMax) / NT * 100
    print(error)
    
    indID = np.repeat(np.arange(N), T * J)
    obsID = np.repeat(np.arange(NT), J)
    altID = np.tile(np.arange(J), NT)
    
    ###
    #Estimate MXL via MCMC
    ###
    
    #xRnd = -np.stack((cost, tt), axis = 1) #np.zeros((0,0)) #-np.hstack((cost, he, tt))
    
    #Random parameter distributions
    #0: normal
    #1: log-normal
    #2: S_B
    xRnd_trans = np.array([0, 0])
    
    zeta_inits = np.zeros((xRnd.shape[1],))
    OmegaB_inits = 0.1 * np.eye(xRnd.shape[1])
    OmegaW_inits = 0.1 * np.eye(xRnd.shape[1])
    
    A = 1.04
    nu = 2
    diagCov = (False, False)
    
    mcmc_nChain = 2
    mcmc_iterBurn = 2000
    mcmc_iterSample = 2000
    mcmc_thin = 10
    mcmc_iterMem = mcmc_iterSample
    mcmc_disp = 1000
    seed = 4711
    simLogLik = True
    simLogLikDrawsType = 'haltonShiftShuffle'
    simDraws = 200  
    
    rho = 0.1
    
    modelName = 'test'
    deleteDraws = True
    
    results = estimate(
            mcmc_nChain, mcmc_iterBurn, mcmc_iterSample, mcmc_thin, mcmc_iterMem, mcmc_disp, 
            seed, simLogLik, simLogLikDrawsType, simDraws,
            rho,
            modelName, deleteDraws,
            A, nu, diagCov,
            zeta_inits, OmegaB_inits, OmegaW_inits,
            indID, obsID, altID, chosen,
            xRnd, xRnd_trans)