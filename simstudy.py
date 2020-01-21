import numpy as np
import scipy.sparse

from mxl import prepareData, pPredMxl

###
#SRMSE
###

def rmse(est, true):
    est = np.ndarray.flatten(est)
    true = np.ndarray.flatten(true)
    s = np.sqrt(np.mean((est - true)**2)) #/ np.absolute(np.mean(true))
    return s

###
#TVD
###

def tvd(pred, true):
    t = np.mean(0.5 * np.absolute(true - pred))
    return t

###
#Predition for MSLE
###
    
def predictionMsle(
        nIter, nTakes, nSim, seed,
        param, iHess,
        indID, obsID, altID, chosen,
        xFix, xRnd):
    
    np.random.seed(seed)
    
    ###
    #Prepare data
    ###
    
    nFix = xFix.shape[1]
    nRnd = xRnd.shape[1]
    
    xList = [xFix, xRnd]
    (xList,
     nInd, nObs, nRow,
     chosenIdx, nonChosenIdx,
     rowsPerInd, rowsPerObs,
     _, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen)
    xFix, xRnd = xList[0], xList[1]
    
    sim_xRnd = np.tile(xRnd, (nSim, 1))
    sim_rowsPerInd = np.tile(rowsPerInd, (nSim,))
    sim_map_avail_to_obs = scipy.sparse.kron(scipy.sparse.eye(nSim), map_avail_to_obs)
    
    chIdx = np.triu_indices(nRnd); chIdx = chIdx[1], chIdx[0];

    ###
    #Prediction
    ###
    
    pPred = np.zeros((nRow + nObs,))
    
    vFix = 0 
    chIHess = np.linalg.cholesky(iHess)
    nParam = param.shape[0]
    
    for i in np.arange(nIter):
        paramSim = param + chIHess @ np.random.randn(nParam,)
        if nFix: 
            paramFixSim = np.array(paramSim[:nFix])
            vFix = np.tile(xFix @ paramFixSim, (nSim,));
        paramRnd_muSim = np.array(paramSim[nFix:(nFix + nRnd)])
        paramRnd_chSim = np.zeros((nRnd, nRnd))
        paramRnd_chSim[chIdx] = np.array(paramSim[(nFix + nRnd):])
        
        pPred_iter = np.zeros((nRow + nObs,))
        
        for t in np.arange(nTakes):
            paramRndSim = paramRnd_muSim + (paramRnd_chSim @ np.random.randn(nRnd, nInd * nSim)).T
            paramRndSimPerRow = np.repeat(paramRndSim, sim_rowsPerInd, axis = 0)
            vRnd = np.sum(sim_xRnd * paramRndSimPerRow, axis = 1)
            
            pPred_take = pPredMxl(vFix, vRnd, sim_map_avail_to_obs, nSim, chosenIdx, nonChosenIdx)
            pPred_iter += pPred_take
        pPred += (pPred_iter / nTakes)
    pPred /= nIter  
    return pPred
    
###
#If main: test
###
    
if __name__ == "__main__":
    A, B = 100, 100
    est = np.random.rand(A,B)
    true = np.random.rand(A,B)
    s = srmse(est,true)
    print(s)
    
