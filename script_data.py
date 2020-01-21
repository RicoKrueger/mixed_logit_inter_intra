import os
import numpy as np
import itertools
import scipy.sparse
import pickle

from mxl import prepareData, pPredMxl

###
#Set seed
###

np.random.seed(4711)

###
#Meta
###

R = 30
S_list = [1, 2] 
N_list = [250, 1000]
T_list = [8, 16]
A = 5

N_val = 25
    
for S, N, T, r in itertools.product(S_list, N_list, T_list, np.arange(R)):
    print(" ")
    print("S = " + str(S) + "; " + 
          "N = " + str(N) + "; " + 
          "T = " + str(T) + "; " + 
          "r = " + str(r) + ";")
    
    ###
    #True parameter values
    ###
    
    nRnd_idx = 1
    nRnd_l = [2, 4, 6]
    nRnd = nRnd_l[nRnd_idx]
    
    r_var_mu = 2
    
    if S in [1,2]:
        pB, pW = 2/3, 1/3  
    else:
        pB, pW = 1/3, 2/3
    
    if S in [1]:
        aB, aW = 0.3, 0.3 #* (2/3)
    else:
        aB, aW = 0.6, 0.6 #* (2/3)

    x_scale_l = [3, 2, 1]
    x_scale = x_scale_l[nRnd_idx]
    
    #Between
    betaRndMu_true_l = np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5])
    betaRndCorrB_true_l = [
            np.array([[0, 1],
                      [1, 0]]),
            np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0]]),
            np.array([[0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 0, 1],
                      [1, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 1],
                      [1, 0, 1, 0, 0, 0],
                      [0, 1, 0, 1, 0, 0]]),
            ]
    
    betaRndMu_true = np.array(betaRndMu_true_l[:nRnd])
    betaRndVarB_true = pB * r_var_mu * np.abs(betaRndMu_true)
    betaRndSdB_true = np.sqrt(betaRndVarB_true)
    betaRndCorrB_true = aB * betaRndCorrB_true_l[nRnd_idx] + np.eye(nRnd)
    betaRndSiB_true = np.diag(betaRndSdB_true) @ betaRndCorrB_true @ np.diag(betaRndSdB_true)
    betaRndChB_true = np.linalg.cholesky(betaRndSiB_true)
    
    #Within
    betaRndCorrW_true_l = [
            np.array([[0, 1],
                      [1, 0]]),
            np.array([[0, 1, 0, 1],
                      [1, 0, 0, 0],
                      [0, 0, 0, 1],
                      [1, 0, 1, 0]]),
            np.array([[0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0]]),
            ]
    
    betaRndVarW_true = pW * r_var_mu * np.abs(betaRndMu_true)
    betaRndSdW_true = np.sqrt(betaRndVarW_true)
    betaRndCorrW_true = aW * betaRndCorrW_true_l[nRnd_idx] + np.eye(nRnd)
    betaRndSiW_true = np.diag(betaRndSdW_true) @ betaRndCorrW_true @ np.diag(betaRndSdW_true)
    betaRndChW_true = np.linalg.cholesky(betaRndSiW_true)
    
    ###
    #Generate semi-synthetic data
    ###
    
    T_val = 1
    NT_val = N_val * T_val
    NTA_val = NT_val * A
    
    N_tot = N + N_val
    
    obsPerInd_tot = np.concatenate((np.repeat(T, N),
                                    np.repeat(1, N_val)))
    obsPerInd_tot[:N_val] += 1
    rowsPerInd_tot = obsPerInd_tot * A
    rowsPerObs_tot = np.repeat(A, np.sum(obsPerInd_tot))
    
    NT_tot = np.sum(obsPerInd_tot)
    NTA_tot = np.sum(rowsPerObs_tot)
    
    indObsID_tot = np.zeros((NTA_tot,))
    u = 0
    for n in np.arange(N_tot):
        indObsID_ind = np.kron(np.arange(1, obsPerInd_tot[n] + 1).reshape((-1,1)), 
                               np.ones((A, 1))).reshape((-1,))
        l = u; u += rowsPerInd_tot[n];
        indObsID_tot[l:u] = indObsID_ind    
    
    
    #Generate parameters    
    betaRndInd_true = betaRndMu_true + (betaRndChB_true @ np.random.randn(nRnd, N_tot)).T
    betaRndObs_true = (betaRndChW_true @ np.random.randn(nRnd, NT_tot)).T
    betaRnd_true = np.repeat(betaRndInd_true, rowsPerInd_tot, axis = 0) + \
    np.repeat(betaRndObs_true, rowsPerObs_tot, axis = 0)
    
    #Sample choice sets
    xRnd_tot = x_scale * np.random.rand(NTA_tot, nRnd)
    
    #Simulate choices
    chosen_tot = np.zeros((NTA_tot,), dtype = 'int64')
    
    eps = -np.log(-np.log(np.random.rand(NTA_tot,)))
    vDet = np.sum(xRnd_tot * betaRnd_true, axis = 1)
    v = vDet + eps
    
    errorChoice = np.empty((NT_tot,), dtype = 'bool')
    
    for i in np.arange(NT_tot):
        sl = slice(i * A, (i + 1) * A)
        choiceDet = np.where(vDet[sl] == vDet[sl].max())
        choiceRnd = np.where(v[sl] == v[sl].max())
        errorChoice[i] = choiceRnd[0] not in choiceDet[0]
        altMax = np.random.choice(choiceRnd[0], size = 1)
        chosen_tot[i * A + altMax] = 1
        
    altID_tot = np.tile(np.arange(1, A + 1), (NT_tot,)) 
        
    error = np.sum(errorChoice) / NT_tot
    print("Error rate: " + str(error))
    
    chosenAlt_tot = chosen_tot * altID_tot
    chosenAlt_tot = np.array(chosenAlt_tot[chosen_tot == 1])
    _, ms = np.unique(chosenAlt_tot, return_counts = True)
    ms = ms / NT_tot
    print("Market shares :")
    print(ms)
    
    #Extract relevant data
    betaRndInd_true = np.array(betaRndInd_true[:N,:])
    
    indID_tot = np.repeat(np.arange(N_tot), rowsPerInd_tot)
    obsID_tot = np.repeat(np.arange(NT_tot), rowsPerObs_tot)

    idx = np.logical_and(indObsID_tot <= T, indID_tot < N) #Training data
    indID = np.array(indID_tot[idx])
    obsID = np.array(obsID_tot[idx])
    indObsID = np.array(indObsID_tot[idx])
    altID = np.array(altID_tot[idx])
    chosen = np.array(chosen_tot[idx])
    xRnd = np.array(xRnd_tot[idx,:])
    
    betaRndObs_true = np.array(betaRndObs_true[np.unique(obsID),:])
    
    idx = indID_tot >= N #Between validation
    indID_valB = np.array(indID_tot[idx])
    obsID_valB = np.array(obsID_tot[idx])
    altID_valB = np.array(altID_tot[idx])
    chosen_valB = np.array(chosen_tot[idx])
    xRnd_valB = np.array(xRnd_tot[idx,:])
    
    idx = indObsID_tot > T #Within validation
    indID_valW = np.array(indID_tot[idx])
    obsID_valW = np.array(obsID_tot[idx])
    altID_valW = np.array(altID_tot[idx])
    chosen_valW = np.array(chosen_tot[idx])
    xRnd_valW = np.array(xRnd_tot[idx,:]) 
    
    #Simulate predictive choice distributions for the between validation sample 
    D_B = 1000; nTakes_B = 2; simDraws_B = D_B * nTakes_B; 
    D_W = 1000; nTakes_W = 2; simDraws_W = D_W * nTakes_W;
    
    xList = [xRnd_valB]
        
    (xList,
     _, _, _,
     chosenIdx, nonChosenIdx,
     rowsPerInd, _,
     _, map_avail_to_obs) = prepareData(xList, indID_valW, obsID_valW, chosen_valW)
    
    xRnd_valDiff = xList[0]
    xRnd_valDiff = np.tile(xRnd_valDiff, (D_B, 1))
    sim_map_avail_to_obs = scipy.sparse.kron(scipy.sparse.eye(D_B), map_avail_to_obs)
    sim_rowsPerInd = np.tile(rowsPerInd, D_B)
    rowsPerObs = np.squeeze(np.asarray(np.sum(map_avail_to_obs, axis = 0)))
    sim_rowsPerObs = np.tile(rowsPerObs, D_B)
    
    pPred_true = np.zeros((NTA_val,))
    
    vFix = 0
    
    for i in np.arange(simDraws_B):
        betaRndInd = betaRndMu_true + (betaRndChB_true @ np.random.randn(nRnd, N_val)).T
        betaRndInd_perRow = np.tile(np.repeat(betaRndInd, rowsPerInd, axis = 0), (D_B, 1))
        
        for j in np.arange(nTakes_B):
            betaRndObs = (betaRndChW_true @ np.random.randn(nRnd, NT_val * D_W)).T
            betaRnd = betaRndInd_perRow + np.repeat(betaRndObs, sim_rowsPerObs, axis = 0)
            vRnd = np.sum(xRnd_valDiff * betaRnd, axis = 1)
            
            pPred_true_take = pPredMxl(vFix, vRnd, sim_map_avail_to_obs, D_W, chosenIdx, nonChosenIdx)
            pPred_true += pPred_true_take 
    pPred_true /= (D_B * nTakes_B**2)
    pPredB_true = np.array(pPred_true)   

    #Simulate predictive choice distributions for the within validation sample 
    D_W = 1000; nTakes_W = 10; simDraws_W = D_W * nTakes_W;
    
    xList = [xRnd_valW]
        
    (xList,
     _, _, _,
     chosenIdx, nonChosenIdx,
     rowsPerInd, _,
     _, map_avail_to_obs) = prepareData(xList, indID_valW, obsID_valW, chosen_valW)
    
    xRnd_valDiff = xList[0]
    xRnd_valDiff = np.tile(xRnd_valDiff, (D_W, 1))
    sim_map_avail_to_obs = scipy.sparse.kron(scipy.sparse.eye(D_W), map_avail_to_obs)
    sim_rowsPerInd = np.tile(rowsPerInd, D_W)
    rowsPerObs = np.squeeze(np.asarray(np.sum(map_avail_to_obs, axis = 0)))
    sim_rowsPerObs = np.tile(rowsPerObs, D_W)
    
    pPred_true = np.zeros((NTA_val,))
    
    vFix = 0    
    betaRndInd_perRow = np.tile(np.repeat(betaRndInd_true[:N_val,:], rowsPerInd, axis = 0), 
                                (D_W, 1))
    
    for i in np.arange(nTakes_W):
        betaRndObs = (betaRndChW_true @ np.random.randn(nRnd, NT_val * D_W)).T
        betaRnd = betaRndInd_perRow + np.repeat(betaRndObs, sim_rowsPerObs, axis = 0)
        vRnd = np.sum(xRnd_valDiff * betaRnd, axis = 1)
        
        pPred_true_take = pPredMxl(vFix, vRnd, sim_map_avail_to_obs, D_W, chosenIdx, nonChosenIdx)
        pPred_true += pPred_true_take 
    pPred_true /= nTakes_W
    pPredW_true = np.array(pPred_true)   
    
    #Save data 
    sim = {'indID': indID, 'obsID': obsID, 'indObsID': indObsID,
           'altID': altID, 'chosen': chosen,
           'xRnd': xRnd, 'nRnd': nRnd,
           'indID_valB': indID_valB, 'obsID_valB': obsID_valB, 
           'altID_valB': altID_valB, 'chosen_valB': chosen_valB,
           'xRnd_valB': xRnd_valB,
           'indID_valW': indID_valW, 'obsID_valW': obsID_valW, 
           'altID_valW': altID_valW, 'chosen_valW': chosen_valW,
           'xRnd_valW': xRnd_valW,
           'betaRndInd_true': betaRndInd_true,
           'betaRndObs_true': betaRndObs_true,
           'pPredB_true': pPredB_true, 'pPredW_true': pPredW_true}
    
    filename = 'sim' + '_S' + str(S) + '_N' + str(N) + '_T' + str(T) + '_r' + str(r)
    if os.path.exists(filename): os.remove(filename) 
    outfile = open(filename, 'wb')
    pickle.dump(sim, outfile)
    outfile.close()
    