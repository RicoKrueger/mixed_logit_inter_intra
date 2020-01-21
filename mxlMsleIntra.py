from joblib import Parallel, delayed
import time
import pandas as pd
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.stats

from mxl import prepareData, transFix, transRnd
from qmc import makeNormalDraws
                
###
#MSLE
###

def transDerFix(derFix, paramFix, xFix_trans):
    derFix_trans = np.array(derFix)
    
    idx = xFix_trans == 1
    if np.sum(idx) > 0:
        derFix_trans[:, idx] = derFix[:, idx] * paramFix[idx]
    return derFix_trans
    
def transDerRnd(derRnd, paramRnd, xRnd_trans):
    derRnd_trans = np.array(derRnd)
    
    idx = xRnd_trans == 1
    if np.sum(idx) > 0:
        derRnd_trans[:, idx] = paramRnd[:, idx]
        
    idx = xRnd_trans == 2
    if np.sum(idx) > 0:
        derRnd_trans[:, idx] = paramRnd[:, idx] - paramRnd[:, idx] ** 2    
    return derRnd_trans

def derRnd(nInd, nDrawsMem, nRnd, xRnd_transBool, paramRnd, xRnd_trans, chIdx, 
           drawsTake, sim_rowsPerInd, uc):
    derRnd = np.ones((nInd * nDrawsMem, nRnd))
    if xRnd_transBool: derRnd = transDerRnd(derRnd, paramRnd, xRnd_trans)
    derRnd_mu_ind = derRnd
    if uc:
        derRnd_ch_ind = derRnd * drawsTake
    else:
        derRnd_ch_ind = derRnd[:, chIdx[0]] * drawsTake[:, chIdx[1]]
    derRnd_mu = np.repeat(derRnd_mu_ind, sim_rowsPerInd, axis = 0)
    derRnd_ch = np.repeat(derRnd_ch_ind, sim_rowsPerInd, axis = 0)
    return derRnd_mu, derRnd_ch

def derRnd_ch(nObs, nDrawsMem, nRnd, xRnd_transBool, paramRnd, xRnd_trans, chIdx, 
           drawsTake, sim_rowsPerObs, uc):
    derRnd = np.ones((nObs * nDrawsMem, nRnd))
    if xRnd_transBool: derRnd = transDerRnd(derRnd, paramRnd, xRnd_trans)
    if uc:
        derRnd_ch_ind = derRnd * drawsTake
    else:
        derRnd_ch_ind = derRnd[:, chIdx[0]] * drawsTake[:, chIdx[1]]
    derRnd_ch = np.repeat(derRnd_ch_ind, sim_rowsPerObs, axis = 0)
    return derRnd_ch

def probGrMxl(
        param,
        sim_xFix, xFix_transBool, xFix_trans, nFix, 
        sim_xRndUc, xRndUc_transBool, xRndUc_trans, nRndUc,
        sim_xRndCo, xRndCo_transBool, xRndCo_trans, nRndCo, chIdx,
        sim_xRnd2Uc, xRnd2Uc_transBool, xRnd2Uc_trans, nRnd2Uc,
        sim_xRnd2Co, xRnd2Co_transBool, xRnd2Co_trans, nRnd2Co, ch2Idx,
        drawsUcTake, drawsCoTake, nDrawsMem,
        drawsUcBTake, drawsCoBTake, nDrawsBMem,
        drawsUcWTake, drawsCoWTake, nDrawsWMem,
        nInd, nObs, 
        simB_rowsPerInd, simB_obsPerInd,
        simW_rowsPerObs,
        simW_map_avail_to_obs, simW_map_drawsB_to_obs):
    ###
    #Utility
    ###
    
    vFix = 0; vRndUc = 0; vRndCo = 0; vRnd2Uc = 0; vRnd2Co = 0;
    u = 0
    if nFix > 0:
        l = u; u += nFix;
        paramFix = np.array(param[l:u])
        if xFix_transBool: paramFix = np.array(transFix(paramFix, xFix_trans))
        vFix = sim_xFix @ paramFix #Good
    if nRndUc > 0:
        l = u; u += nRndUc;
        paramRndUc_mu = np.array(param[l:u])
        l = u; u += nRndUc;
        paramRndUc_sd = np.array(param[l:u])
        paramRndUc = paramRndUc_mu + paramRndUc_sd * drawsUcTake
        if xRndUc_transBool: paramRndUc = np.array(transRnd(paramRndUc, xRndUc_trans))
        paramRndUcPerRow = np.tile(np.repeat(paramRndUc, simB_rowsPerInd, axis = 0), (nDrawsWMem, 1)) #Good
        vRndUc = np.sum(sim_xRndUc * paramRndUcPerRow, axis = 1)       
    if nRndCo > 0:
        l = u; u += nRndCo;
        paramRndCo_mu = np.array(param[l:u])
        l = u; u += int((nRndCo * (nRndCo + 1)) / 2)
        paramRndCo_ch = np.zeros((nRndCo, nRndCo))
        paramRndCo_ch[chIdx] = np.array(param[l:u])
        paramRndCo = paramRndCo_mu + (paramRndCo_ch @ drawsCoTake.T).T
        if xRndCo_transBool: paramRndCo = np.array(transRnd(paramRndCo, xRndCo_trans))
        paramRndCoPerRow = np.tile(np.repeat(paramRndCo, simB_rowsPerInd, axis = 0), (nDrawsWMem, 1)) #Good
        vRndCo = np.sum(sim_xRndCo * paramRndCoPerRow, axis = 1)
    if nRnd2Uc > 0:
        l = u; u += nRnd2Uc;
        paramRnd2Uc_mu = np.array(param[l:u])
        l = u; u += nRnd2Uc;
        paramRnd2Uc_sdB = np.array(param[l:u])
        l = u; u += nRnd2Uc;
        paramRnd2Uc_sdW = np.array(param[l:u])
        paramRnd2Uc_ind = paramRnd2Uc_mu + paramRnd2Uc_sdB * drawsUcBTake #Good
        paramRnd2Uc_obs = paramRnd2Uc_sdW * drawsUcWTake #Good
        paramRnd2Uc = np.tile(np.repeat(paramRnd2Uc_ind, simB_obsPerInd, axis = 0), (nDrawsWMem, 1)) + paramRnd2Uc_obs #Good
        if xRnd2Uc_transBool: paramRnd2Uc = np.array(transRnd(paramRnd2Uc, xRnd2Uc_trans))
        paramRnd2UcPerRow = np.repeat(paramRnd2Uc, simW_rowsPerObs, axis = 0) #Good
        vRnd2Uc = np.sum(sim_xRnd2Uc * paramRnd2UcPerRow, axis = 1)       
    if nRnd2Co > 0:
        l = u; u += nRnd2Co;
        paramRnd2Co_mu = np.array(param[l:u])
        l = u; u += int((nRnd2Co * (nRnd2Co + 1)) / 2)
        paramRnd2Co_chB = np.zeros((nRnd2Co, nRnd2Co))
        paramRnd2Co_chB[ch2Idx] = np.array(param[l:u])
        l = u; u += int((nRnd2Co * (nRnd2Co + 1)) / 2)
        paramRnd2Co_chW = np.zeros((nRnd2Co, nRnd2Co))
        paramRnd2Co_chW[ch2Idx] = np.array(param[l:u])
        paramRnd2Co_ind = paramRnd2Co_mu + (paramRnd2Co_chB @ drawsCoBTake.T).T #Good
        paramRnd2Co_obs = (paramRnd2Co_chW @ drawsCoWTake.T).T #Good
        paramRnd2Co = np.tile(np.repeat(paramRnd2Co_ind, simB_obsPerInd, axis = 0), (nDrawsWMem, 1)) + paramRnd2Co_obs #Good
        if xRnd2Co_transBool: paramRnd2Co = np.array(transRnd(paramRnd2Co, xRnd2Co_trans))
        paramRnd2CoPerRow = np.repeat(paramRnd2Co, simW_rowsPerObs, axis = 0) #Good
        vRnd2Co = np.sum(sim_xRnd2Co * paramRnd2CoPerRow, axis = 1)        
        
    v = vFix + vRndUc + vRndCo + vRnd2Uc + vRnd2Co
    
    ###
    #Probability
    ###
    
    ev = np.exp(v)
    ev[ev > 1e+200] = 1e+200
    ev[ev < 1e-200] = 1e-200 
    nev = simW_map_avail_to_obs.T @ ev + 1
    nnev = simW_map_avail_to_obs @ nev;
    sim_pChosen = 1 / nev
    sim_pChosen[sim_pChosen < 1e-200] = 1e-200
    sim_pNonChosen = ev / nnev
    sim_pNonChosen[sim_pNonChosen < 1e-200] = 1e-200
    pChosen = sim_pChosen.reshape((nDrawsWMem, nDrawsBMem, nObs)).sum(axis = 0).reshape((-1,1))
    
    ###
    #Gradient
    ###
    
    def calcGradient(der):
        t1 = -sim_pNonChosen.reshape((-1,1)) * der
        t2 = simW_map_avail_to_obs.T @ t1
        numer = sim_pChosen.reshape((-1,1)) * t2
        sgr = simW_map_drawsB_to_obs @ numer
        return sgr
    
    sgrFix = np.empty((nObs * nDrawsBMem, 0))
    sgrRndUc_mu = np.empty((nObs * nDrawsBMem, 0))
    sgrRndUc_sd = np.empty((nObs * nDrawsBMem, 0))
    sgrRndCo_mu = np.empty((nObs * nDrawsBMem, 0))
    sgrRndCo_ch = np.empty((nObs * nDrawsBMem, 0))
    sgrRnd2Uc_mu = np.empty((nObs * nDrawsBMem, 0))
    sgrRnd2Uc_sdB = np.empty((nObs * nDrawsBMem, 0))
    sgrRnd2Uc_sdW = np.empty((nObs * nDrawsBMem, 0))
    sgrRnd2Co_mu = np.empty((nObs * nDrawsBMem, 0))
    sgrRnd2Co_chB = np.empty((nObs * nDrawsBMem, 0))
    sgrRnd2Co_chW = np.empty((nObs * nDrawsBMem, 0))
    
    if nFix > 0:
        derFix = sim_xFix
        if xFix_transBool: derFix = np.array(transDerFix(derFix, paramFix, xFix_trans))
        sgrFix = calcGradient(derFix)
    
    if nRndUc > 0:
        derRndUc_mu, derRndUc_sd = derRnd(nInd, nDrawsBMem, nRndUc, 
                                          xRndUc_transBool, paramRndUc, xRndUc_trans, 
                                          chIdx, drawsUcTake, simB_rowsPerInd, True)
        derRndUc_mu = np.tile(derRndUc_mu, (nDrawsWMem, 1))
        derRndUc_sd = np.tile(derRndUc_sd, (nDrawsWMem, 1))
        derRndUc_mu *= sim_xRndUc
        derRndUc_sd *= sim_xRndUc
        sgrRndUc_mu = calcGradient(derRndUc_mu)
        sgrRndUc_sd = calcGradient(derRndUc_sd)    
    
    if nRndCo > 0:
        derRndCo_mu, derRndCo_ch = derRnd(nInd, nDrawsBMem, nRndCo, 
                                          xRndCo_transBool, paramRndCo, xRndCo_trans, 
                                          chIdx, drawsCoTake, simB_rowsPerInd, False)
        derRndCo_mu = np.tile(derRndCo_mu, (nDrawsWMem, 1))
        derRndCo_ch = np.tile(derRndCo_ch, (nDrawsWMem, 1))
        derRndCo_mu *= sim_xRndCo
        derRndCo_ch *= sim_xRndCo[:, chIdx[0]]
        sgrRndCo_mu = calcGradient(derRndCo_mu)
        sgrRndCo_ch = calcGradient(derRndCo_ch) 
        
    if nRnd2Uc > 0:
        derRnd2Uc_mu, derRnd2Uc_sdB = derRnd(nInd, nDrawsBMem, nRnd2Uc, 
                                             xRnd2Uc_transBool, paramRnd2Uc_ind, xRnd2Uc_trans, 
                                             ch2Idx, drawsUcBTake, simB_rowsPerInd, True)
        derRnd2Uc_mu = np.tile(derRnd2Uc_mu, (nDrawsWMem, 1))
        derRnd2Uc_sdB = np.tile(derRnd2Uc_sdB, (nDrawsWMem, 1))
        derRnd2Uc_sdW = derRnd_ch(nObs, nDrawsMem, nRnd2Uc, 
                               xRnd2Uc_transBool, paramRnd2Uc_obs, xRnd2Uc_trans, 
                               ch2Idx, drawsUcBTake, simW_rowsPerObs, True)
        derRnd2Uc_mu *= sim_xRnd2Uc
        derRnd2Uc_sdB *= sim_xRnd2Uc
        derRnd2Uc_sdW *= sim_xRnd2Uc
        sgrRnd2Uc_mu = calcGradient(derRnd2Uc_mu)
        sgrRnd2Uc_sdB = calcGradient(derRnd2Uc_sdB) 
        sgrRnd2Uc_sdW = calcGradient(derRnd2Uc_sdW)  
        
    if nRnd2Co > 0:
        derRnd2Co_mu, derRnd2Co_chB = derRnd(nInd, nDrawsBMem, nRnd2Co, 
                                             xRnd2Co_transBool, paramRnd2Co_ind, xRnd2Co_trans, 
                                             ch2Idx, drawsCoBTake, simB_rowsPerInd, False)
        derRnd2Co_mu = np.tile(derRnd2Co_mu, (nDrawsWMem, 1))
        derRnd2Co_chB = np.tile(derRnd2Co_chB, (nDrawsWMem, 1))
        derRnd2Co_chW = derRnd_ch(nObs, nDrawsMem, nRnd2Co, 
                               xRnd2Co_transBool, paramRnd2Co_obs, xRnd2Co_trans, 
                               ch2Idx, drawsCoWTake, simW_rowsPerObs, False)
        derRnd2Co_mu *= sim_xRnd2Co
        derRnd2Co_chB *= sim_xRnd2Co[:, ch2Idx[0]]
        derRnd2Co_chW *= sim_xRnd2Co[:, ch2Idx[0]]
        sgrRnd2Co_mu = calcGradient(derRnd2Co_mu)
        sgrRnd2Co_chB = calcGradient(derRnd2Co_chB) 
        sgrRnd2Co_chW = calcGradient(derRnd2Co_chW)          

    sgr = np.concatenate((sgrFix, 
                          sgrRndUc_mu, sgrRndUc_sd, 
                          sgrRndCo_mu, sgrRndCo_ch,
                          sgrRnd2Uc_mu, sgrRnd2Uc_sdB, sgrRnd2Uc_sdW,
                          sgrRnd2Co_mu, sgrRnd2Co_chB, sgrRnd2Co_chW), axis = 1) 
    
    return pChosen, sgr

def objectiveMxl_batch(
        param,
        sim_xFix, xFix_transBool, xFix_trans, nFix, 
        sim_xRndUc, xRndUc_transBool, xRndUc_trans, nRndUc,
        sim_xRndCo, xRndCo_transBool, xRndCo_trans, nRndCo, chIdx,
        sim_xRnd2Uc, xRnd2Uc_transBool, xRnd2Uc_trans, nRnd2Uc,
        sim_xRnd2Co, xRnd2Co_transBool, xRnd2Co_trans, nRnd2Co, ch2Idx,
        drawsUc, drawsCo, nDrawsMem,
        drawsUcB, drawsCoB, nDrawsBMem, nTakesB,
        drawsUcW, drawsCoW, nDrawsW, nDrawsWMem, nTakesW,
        nInd, nObs,
        simB_rowsPerInd, simB_obsPerInd, 
        simW_rowsPerObs,
        simB_map_obs_to_ind, simB_map_draws_to_ind,
        simW_map_avail_to_obs, simW_map_drawsB_to_obs):
    
    pInd = 0
    sgr = 0
       
    for tB in np.arange(nTakesB):
        if nRndUc > 0 and nTakesB > 1:
            drawsUcTake = drawsUc[tB,:,:]
        else:
            drawsUcTake = drawsUc  
        if nRndCo > 0 and nTakesB > 1:
            drawsCoTake = drawsCo[tB,:,:]   
        else:
            drawsCoTake = drawsCo 
        if nRnd2Uc > 0 and nTakesB > 1:
            drawsUcBTake = drawsUcB[tB,:,:]
        else:
            drawsUcBTake = drawsUcB  
        if nRnd2Co > 0 and nTakesB > 1:
            drawsCoBTake = drawsCoB[tB,:,:] 
        else:
            drawsCoBTake = drawsCoB
            
        pChosen_sim = 0
        sgr_sim = 0
        
        for tW in np.arange(nTakesW):
            if nRnd2Uc > 0 and nTakesW > 1:
                drawsUcWTake = drawsUcW[tW,:,:]
            else:
                drawsUcWTake = drawsUcW  
            if nRnd2Co > 0 and nTakesB > 1:
                drawsCoWTake = drawsCoW[tW,:,:] 
            else:
                drawsCoWTake = drawsCoW            
        
            pChosen_take, sgr_take = probGrMxl(
                    param,
                    sim_xFix, xFix_transBool, xFix_trans, nFix, 
                    sim_xRndUc, xRndUc_transBool, xRndUc_trans, nRndUc,
                    sim_xRndCo, xRndCo_transBool, xRndCo_trans, nRndCo, chIdx,
                    sim_xRnd2Uc, xRnd2Uc_transBool, xRnd2Uc_trans, nRnd2Uc,
                    sim_xRnd2Co, xRnd2Co_transBool, xRnd2Co_trans, nRnd2Co, ch2Idx,
                    drawsUcTake, drawsCoTake, nDrawsMem,
                    drawsUcBTake, drawsCoBTake, nDrawsBMem,
                    drawsUcWTake, drawsCoWTake, nDrawsWMem,
                    nInd, nObs, 
                    simB_rowsPerInd, simB_obsPerInd,
                    simW_rowsPerObs,
                    simW_map_avail_to_obs, simW_map_drawsB_to_obs)
            
            pChosen_sim += pChosen_take
            sgr_sim += sgr_take
            
        pChosen_sim /= nDrawsW
        sgr_sim /= nDrawsW
        
        #Likelihood
        lPChosen_sim = np.log(pChosen_sim)
        pInd_sim = np.exp(simB_map_obs_to_ind.T @ lPChosen_sim)
        pInd_take = pInd_sim.reshape((nDrawsBMem, nInd)).sum(axis = 0).reshape((nInd,1))
        
        #Gradient
        sgr_sim /= pChosen_sim
        sgr_sum_sim = simB_map_obs_to_ind.T @ sgr_sim
        sgr_d_sim = pInd_sim * sgr_sum_sim
        sgr_take = simB_map_draws_to_ind @ sgr_d_sim
        
        pInd += pInd_take
        sgr += sgr_take
    
    return pInd, sgr        


def objectiveMxl(
        param, return_bhhh, K,
        sim_xFix, xFix_transBool, xFix_trans, nFix, 
        sim_xRndUc, xRndUc_transBool, xRndUc_trans, nRndUc,
        sim_xRndCo, xRndCo_transBool, xRndCo_trans, nRndCo, chIdx,
        sim_xRnd2Uc, xRnd2Uc_transBool, xRnd2Uc_trans, nRnd2Uc,
        sim_xRnd2Co, xRnd2Co_transBool, xRnd2Co_trans, nRnd2Co, ch2Idx,
        drawsUc, drawsCo, nDrawsMem,
        drawsUcB, drawsCoB, nDrawsB, nDrawsBMem, nTakesB,
        drawsUcW, drawsCoW, nDrawsW, nDrawsWMem, nTakesW,
        nInd, nObs,
        simB_rowsPerInd, simB_obsPerInd, 
        simW_rowsPerObs,
        simB_map_obs_to_ind, simB_map_draws_to_ind,
        simW_map_avail_to_obs, simW_map_drawsB_to_obs):
    
    aux = Parallel(n_jobs = K)(delayed(objectiveMxl_batch)(
            param,
            sim_xFix, xFix_transBool, xFix_trans, nFix, 
            sim_xRndUc, xRndUc_transBool, xRndUc_trans, nRndUc,
            sim_xRndCo, xRndCo_transBool, xRndCo_trans, nRndCo, chIdx,
            sim_xRnd2Uc, xRnd2Uc_transBool, xRnd2Uc_trans, nRnd2Uc,
            sim_xRnd2Co, xRnd2Co_transBool, xRnd2Co_trans, nRnd2Co, ch2Idx,
            drawsUc[k,:,:,:], drawsCo[k,:,:,:], nDrawsMem,
            drawsUcB[k,:,:,:], drawsCoB[k,:,:,:], nDrawsBMem, nTakesB,
            drawsUcW, drawsCoW, nDrawsW, nDrawsWMem, nTakesW,
            nInd, nObs,
            simB_rowsPerInd, simB_obsPerInd, 
            simW_rowsPerObs,
            simB_map_obs_to_ind, simB_map_draws_to_ind,
            simW_map_avail_to_obs, simW_map_drawsB_to_obs)
    for k in np.arange(K))
        
    pInd = np.stack([aux[k][0] for k in np.arange(K)], axis=0).sum(axis=0)
    sgr = np.stack([aux[k][1] for k in np.arange(K)], axis=0).sum(axis=0) 
    
    sgr /= pInd
    pInd /= nDrawsB
    
    ll = -np.sum(np.log(pInd), axis = 0)
    gr = -np.sum(sgr, axis = 0)
    
    if return_bhhh:
        bhhh = sgr.T @ sgr
        return ll, gr, bhhh
    else:
        return ll, gr

###
#Process output
###
 
def processOutput(est, se, zVal, pVal, lu):
    colHeaders = ['est.', 'std. err.', 'z-val.', 'p-val.']
    param_est = est[lu]
    param_se = se[lu]
    param_zVal = zVal[lu]
    param_pVal = pVal[lu]
    pd_param = pd.DataFrame(np.stack((param_est, param_se, param_zVal, param_pVal), axis = 1), columns = colHeaders)
    print(pd_param)
    return param_est, param_se, param_zVal, param_pVal, pd_param

###
#Estimate
###
    
def estimate(
        drawsType, nDrawsB, nTakesB, nDrawsW, nTakesW, K,
        seed, modelName, deleteDraws,
        paramFix_inits, 
        paramRndUc_mu_inits, paramRndUc_sd_inits, 
        paramRndCo_mu_inits, paramRndCo_ch_inits,
        paramRnd2Uc_mu_inits, paramRnd2Uc_sdB_inits, paramRnd2Uc_sdW_inits, 
        paramRnd2Co_mu_inits, paramRnd2Co_chB_inits, paramRnd2Co_chW_inits,
        indID, obsID, altID, chosen,
        xFix, xRndUc, xRndCo, xRnd2Uc, xRnd2Co,
        xFix_trans, xRndUc_trans, xRndCo_trans, xRnd2Uc_trans, xRnd2Co_trans):
    
    np.random.seed(seed)
    
    ###
    #Prepare data
    ###
    
    nFix = xFix.shape[1]
    nRndUc = xRndUc.shape[1]
    nRndCo = xRndCo.shape[1]
    nRnd = nRndUc + nRndCo
    nRnd2Uc = xRnd2Uc.shape[1]
    nRnd2Co = xRnd2Co.shape[1]
    nRnd2 = nRnd2Uc + nRnd2Co
    
    if nRnd == 0 and nRnd2 == 0:
        nDrawsB = 1
        nDrawsBMem = 1
        nTakesB = 1
        
        nDrawsW = 1
        nDrawsWMem = 1
        nTakesW = 1
    if nRnd > 0 and nRnd2 == 0:
        K = 1
        nDrawsBMem, mod = divmod(nDrawsB, nTakesB)
        assert mod == 0, "nDrawsB is not multiple of nTakesB!"
        nDrawsW, nDrawsWMem, nTakesW = 1, 1, 1
    if nRnd2 > 0:
        #If there are random parameters with intra-individual heterogeneity,
        #nDrawsBMem is necessarily one
        nDrawsBMem = 1
        nTakesB = int(nDrawsB / K)
        nDrawsWMem, mod = divmod(nDrawsW, nTakesW)
        assert mod == 0, "nDrawsW is not multiple of nTakesW!"
        
    nDrawsMem = nDrawsBMem * nDrawsWMem
        
    xFix_transBool = np.sum(xFix_trans) > 0
    xRndUc_transBool = np.sum(xRndUc_trans) > 0 
    xRndCo_transBool = np.sum(xRndCo_trans) > 0 
    xRnd2Uc_transBool = np.sum(xRnd2Uc_trans) > 0 
    xRnd2Co_transBool = np.sum(xRnd2Co_trans) > 0 
    
    xList = [xFix, xRndUc, xRndCo, xRnd2Uc, xRnd2Co]
    (xList,
     nInd, nObs, nRow,
     chosenIdx, nonChosenIdx,
     rowsPerInd, rowsPerObs,
     map_obs_to_ind, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen)
    xFix, xRndUc, xRndCo, xRnd2Uc, xRnd2Co = xList[0], xList[1], xList[2], xList[3], xList[4]
    
    sim_xFix = np.tile(xFix, (nDrawsMem, 1))
    sim_xRndUc = np.tile(xRndUc, (nDrawsMem, 1))
    sim_xRndCo = np.tile(xRndCo, (nDrawsMem, 1))
    sim_xRnd2Uc = np.tile(xRnd2Uc, (nDrawsMem, 1))
    sim_xRnd2Co = np.tile(xRnd2Co, (nDrawsMem, 1)) #Good
    
    obsPerInd = np.array(map_obs_to_ind.sum(axis = 0)).reshape((-1,))
    simB_obsPerInd = np.tile(obsPerInd, (nDrawsBMem,))
    simB_rowsPerInd = np.tile(rowsPerInd, (nDrawsBMem,))
    simW_rowsPerObs = np.tile(rowsPerObs, (nDrawsWMem,)) #Good
    
    simB_map_obs_to_ind = scipy.sparse.kron(scipy.sparse.eye(nDrawsBMem), map_obs_to_ind)
    simW_map_avail_to_obs = scipy.sparse.kron(scipy.sparse.eye(nDrawsMem), map_avail_to_obs)
    simB_map_draws_to_ind = scipy.sparse.hstack([scipy.sparse.eye(nInd) for i in np.arange(nDrawsBMem)])
    
    map_obs_to_obs = scipy.sparse.eye(nObs)
    map_aux = scipy.sparse.csr_matrix(np.ones((nDrawsWMem, 1)), dtype = 'int64')
    simW_map_drawsW_to_obs = scipy.sparse.kron(map_aux, map_obs_to_obs)
    simW_map_drawsB_to_obs = scipy.sparse.kron(np.eye(nDrawsBMem), simW_map_drawsW_to_obs).T
    
    chIdx = None
    if nRndCo: 
        chIdx = np.triu_indices(nRndCo); chIdx = chIdx[1], chIdx[0];
        
    ch2Idx = None
    if nRnd2Co: 
        ch2Idx = np.triu_indices(nRnd2Co); ch2Idx = ch2Idx[1], ch2Idx[0];    
             
    ### 
    #Generate draws
    ###
    
    drawsUc = np.zeros((K,1,1,1));
    drawsCo = np.zeros((K,1,1,1)); 
    drawsUcB = np.zeros((K,1,1,1)); drawsUcW = np.zeros((K,1,1,1))
    drawsCoB = np.zeros((K,1,1,1)); drawsCoW = np.zeros((K,1,1,1));
    if nRndUc: 
        _, drawsUc = makeNormalDraws(nDrawsB, nRndUc, drawsType, nInd)
        drawsUc = drawsUc.reshape((K, nTakesB, int(nInd*nDrawsB/K/nTakesB), nRndCo))
    if nRndCo: 
        _, drawsCo = makeNormalDraws(nDrawsB, nRndCo, drawsType, nInd)
        drawsCo = drawsCo.reshape((K, nTakesB, int(nInd*nDrawsB/K/nTakesB), nRndCo))
    if nRnd2Uc: 
        _, drawsUcB = makeNormalDraws(nDrawsB, nRnd2Uc, drawsType, nInd)
        _, drawsUcW = makeNormalDraws(nDrawsW, nRnd2Uc, drawsType, nObs)        
        drawsUcB = drawsUcB.reshape((K, nTakesB, int(nInd*nDrawsB/K/nTakesB), nRnd2Uc))
        drawsUcW = drawsUcW.reshape((nTakesW, int(nObs*nDrawsW/nTakesW), nRnd2Uc))
    if nRnd2Co: 
        _, drawsCoB = makeNormalDraws(nDrawsB, nRnd2Co, drawsType, nInd)
        _, drawsCoW = makeNormalDraws(nDrawsW, nRnd2Co, drawsType, nObs)    
        drawsCoB = drawsCoB.reshape((K, nTakesB, int(nInd*nDrawsB/K/nTakesB), nRnd2Co))
        drawsCoW = drawsCoW.reshape((nTakesW, int(nObs*nDrawsW/nTakesW), nRnd2Co))
    
    ### 
    #Optimise
    ###
    
    paramRndCo_chVec_inits = np.ndarray.flatten(paramRndCo_ch_inits[chIdx])
    paramRnd2Co_chBVec_inits = np.ndarray.flatten(paramRnd2Co_chB_inits[ch2Idx])
    paramRnd2Co_chWVec_inits = np.ndarray.flatten(paramRnd2Co_chW_inits[ch2Idx])
    inits = np.concatenate((paramFix_inits, 
                            paramRndUc_mu_inits, paramRndUc_sd_inits, 
                            paramRndCo_mu_inits, paramRndCo_chVec_inits,
                            paramRnd2Uc_mu_inits, paramRnd2Uc_sdB_inits, paramRnd2Uc_sdW_inits, 
                            paramRnd2Co_mu_inits, paramRnd2Co_chBVec_inits, paramRnd2Co_chWVec_inits), axis = 0)
    
    tic = time.time()
    algo = 'L-BFGS-B'
    resOpt = sp.optimize.minimize(
            fun = objectiveMxl,
            x0 = inits,
            args = (False, K,
                    sim_xFix, xFix_transBool, xFix_trans, nFix, 
                    sim_xRndUc, xRndUc_transBool, xRndUc_trans, nRndUc,
                    sim_xRndCo, xRndCo_transBool, xRndCo_trans, nRndCo, chIdx,
                    sim_xRnd2Uc, xRnd2Uc_transBool, xRnd2Uc_trans, nRnd2Uc,
                    sim_xRnd2Co, xRnd2Co_transBool, xRnd2Co_trans, nRnd2Co, ch2Idx,
                    drawsUc, drawsCo, nDrawsMem,
                    drawsUcB, drawsCoB, nDrawsB, nDrawsBMem, nTakesB,
                    drawsUcW, drawsCoW, nDrawsW, nDrawsWMem, nTakesW,
                    nInd, nObs,
                    simB_rowsPerInd, simB_obsPerInd, 
                    simW_rowsPerObs,
                    simB_map_obs_to_ind, simB_map_draws_to_ind,
                    simW_map_avail_to_obs, simW_map_drawsB_to_obs),
            method = algo,
            jac = True,
            options = {'disp': True})
    toc = time.time() - tic
    
    print(' ')
    print('Computation time [s]: ' + str(toc))
    
    ###
    #Process output
    ###
    
    logLik = -resOpt['fun']
    est = resOpt['x']
    
    if algo == 'BFGS':
        iHess = resOpt['hess_inv']
    if algo == 'L-BFGS-B':
        ll, gr, bhhh = objectiveMxl(
                est, True, K,
                sim_xFix, xFix_transBool, xFix_trans, nFix, 
                sim_xRndUc, xRndUc_transBool, xRndUc_trans, nRndUc,
                sim_xRndCo, xRndCo_transBool, xRndCo_trans, nRndCo, chIdx,
                sim_xRnd2Uc, xRnd2Uc_transBool, xRnd2Uc_trans, nRnd2Uc,
                sim_xRnd2Co, xRnd2Co_transBool, xRnd2Co_trans, nRnd2Co, ch2Idx,
                drawsUc, drawsCo, nDrawsMem,
                drawsUcB, drawsCoB, nDrawsB, nDrawsBMem, nTakesB,
                drawsUcW, drawsCoW, nDrawsW, nDrawsWMem, nTakesW,
                nInd, nObs,
                simB_rowsPerInd, simB_obsPerInd, 
                simW_rowsPerObs,
                simB_map_obs_to_ind, simB_map_draws_to_ind,
                simW_map_avail_to_obs, simW_map_drawsB_to_obs)
        iHess = np.linalg.inv(bhhh)
        
    se = np.sqrt(np.diag(iHess))
    zVal = est / se
    pVal = 2 * scipy.stats.norm.cdf(-np.absolute(zVal))

    u = 0
    if nFix > 0:
        l = u; u += nFix; lu = slice(l,u)
        print(' ')
        print('Fixed parameters:')
        paramFix_est, paramFix_se, paramFix_zVal, paramFix_pVal, pd_paramFix = processOutput(est, se, zVal, pVal, lu)
    else:
        paramFix_est, paramFix_se, paramFix_zVal, paramFix_pVal, pd_paramFix = None, None, None, None, None
        
    if nRndUc > 0:
        l = u; u += nRndUc; lu = slice(l,u)
        print(' ')
        print('Uncorrelated random parameters (means):')
        paramRndUc_mu_est, paramRndUc_mu_se, paramRndUc_mu_zVal, paramRndUc_mu_pVal, pd_paramRndUc_mu = processOutput(est, se, zVal, pVal, lu)
        
        l = u; u += nRndUc; lu = slice(l,u)
        print(' ')
        print('Uncorrelated random parameters (standard deviations):')
        paramRndUc_sd_est, paramRndUc_sd_se, paramRndUc_sd_zVal, paramRndUc_sd_pVal, pd_paramRndUc_sd = processOutput(est, se, zVal, pVal, lu) 
    else:
        paramRndUc_mu_est, paramRndUc_mu_se, paramRndUc_mu_zVal, paramRndUc_mu_pVal, pd_paramRndUc_mu = None, None, None, None, None
        paramRndUc_sd_est, paramRndUc_sd_se, paramRndUc_sd_zVal, paramRndUc_sd_pVal, pd_paramRndUc_sd = None, None, None, None, None
            
    if nRndCo > 0:
        l = u; u += nRndCo; lu = slice(l,u)
        print(' ')
        print('Correlated random parameters (means):')        
        paramRndCo_mu_est, paramRndCo_mu_se, paramRndCo_mu_zVal, paramRndCo_mu_pVal, pd_paramRndCo_mu = processOutput(est, se, zVal, pVal, lu)
        
        l = u; u += int((nRndCo * (nRndCo + 1)) / 2); lu = slice(l,u)
        print(' ')
        print('Correlated random parameters (Cholesky):')
        paramRndCo_ch_est_vec, paramRndCo_ch_se_vec, paramRndCo_ch_zVal_vec, paramRndCo_ch_pVal_vec, pd_paramRndCo_ch = processOutput(est, se, zVal, pVal, lu) 

        print(' ')
        print('Correlated random parameters (Cholesky, est.):')
        paramRndCo_ch_est = np.zeros((nRndCo, nRndCo))
        paramRndCo_ch_est[chIdx] = paramRndCo_ch_est_vec
        print(pd.DataFrame(paramRndCo_ch_est))
        
        print(' ')
        print('Correlated random parameters (Cholesky, std. err.):')
        paramRndCo_ch_se = np.zeros((nRndCo, nRndCo))
        paramRndCo_ch_se[chIdx] = paramRndCo_ch_se_vec
        print(pd.DataFrame(paramRndCo_ch_se))
        
        print(' ')
        print('Correlated random parameters (Cholesky, p-val.):')
        paramRndCo_ch_pVal = np.zeros((nRndCo, nRndCo))
        paramRndCo_ch_pVal[chIdx] = paramRndCo_ch_pVal_vec  
        print(pd.DataFrame(paramRndCo_ch_pVal))
    else:
        paramRndCo_mu_est, paramRndCo_mu_se, paramRndCo_mu_zVal, paramRndCo_mu_pVal, pd_paramRndCo_mu = None, None, None, None, None
        paramRndCo_ch_est, paramRndCo_ch_se, paramRndCo_ch_pVal, pd_paramRndCo_ch = None, None, None, None
        
    if nRnd2Uc > 0:
        l = u; u += nRnd2Uc; lu = slice(l,u)
        print(' ')
        print('Uncorrelated random parameters - intra (means):')
        paramRnd2Uc_mu_est, paramRnd2Uc_mu_se, paramRnd2Uc_mu_zVal, paramRnd2Uc_mu_pVal, pd_paramRnd2Uc_mu = processOutput(est, se, zVal, pVal, lu)
        
        l = u; u += nRnd2Uc; lu = slice(l,u)
        print(' ')
        print('Uncorrelated random parameters - intra-between (standard deviations):')
        paramRnd2Uc_sdB_est, paramRnd2Uc_sdB_se, paramRnd2Uc_sdB_zVal, paramRnd2Uc_sdB_pVal, pd_paramRnd2Uc_sdB = processOutput(est, se, zVal, pVal, lu) 
        
        l = u; u += nRnd2Uc; lu = slice(l,u)
        print(' ')
        print('Uncorrelated random parameters - intra-within (standard deviations):')
        paramRnd2Uc_sdW_est, paramRnd2Uc_sdW_se, paramRnd2Uc_sdW_zVal, paramRnd2Uc_sdW_pVal, pd_paramRnd2Uc_sdW = processOutput(est, se, zVal, pVal, lu) 
    else:
        paramRnd2Uc_mu_est, paramRnd2Uc_mu_se, paramRnd2Uc_mu_zVal, paramRnd2Uc_mu_pVal, pd_paramRnd2Uc_mu = None, None, None, None, None
        paramRnd2Uc_sdB_est, paramRnd2Uc_sdB_se, paramRnd2Uc_sdB_zVal, paramRnd2Uc_sdB_pVal, pd_paramRnd2Uc_sdB = None, None, None, None, None
        paramRnd2Uc_sdW_est, paramRnd2Uc_sdW_se, paramRnd2Uc_sdW_zVal, paramRnd2Uc_sdW_pVal, pd_paramRnd2Uc_sdW = None, None, None, None, None
        
    if nRnd2Co > 0:
        l = u; u += nRnd2Co; lu = slice(l,u)
        print(' ')
        print('Correlated random parameters - intra (means):')        
        paramRnd2Co_mu_est, paramRnd2Co_mu_se, paramRnd2Co_mu_zVal, paramRnd2Co_mu_pVal, pd_paramRnd2Co_mu = processOutput(est, se, zVal, pVal, lu)
        
        l = u; u += int((nRnd2Co * (nRnd2Co + 1)) / 2); lu = slice(l,u)
        print(' ')
        print('Correlated random parameters - intra-between (Cholesky):')
        paramRnd2Co_chB_est_vec, paramRnd2Co_chB_se_vec, paramRnd2Co_chB_zVal_vec, paramRnd2Co_chB_pVal_vec, pd_paramRnd2Co_chB = processOutput(est, se, zVal, pVal, lu) 

        print(' ')
        print('Correlated random parameters - intra-between (Cholesky, est.):')
        paramRnd2Co_chB_est = np.zeros((nRnd2Co, nRnd2Co))
        paramRnd2Co_chB_est[ch2Idx] = paramRnd2Co_chB_est_vec
        print(pd.DataFrame(paramRnd2Co_chB_est))
        
        print(' ')
        print('Correlated random parameters - intra-between (Cholesky, std. err.):')
        paramRnd2Co_chB_se = np.zeros((nRnd2Co, nRnd2Co))
        paramRnd2Co_chB_se[ch2Idx] = paramRnd2Co_chB_se_vec
        print(pd.DataFrame(paramRnd2Co_chB_se))
        
        print(' ')
        print('Correlated random parameters - intra-between (Cholesky, p-val.):')
        paramRnd2Co_chB_pVal = np.zeros((nRnd2Co, nRnd2Co))
        paramRnd2Co_chB_pVal[ch2Idx] = paramRnd2Co_chB_pVal_vec  
        print(pd.DataFrame(paramRnd2Co_chB_pVal))
        
        l = u; u += int((nRnd2Co * (nRnd2Co + 1)) / 2); lu = slice(l,u)
        print(' ')
        print('Correlated random parameters - intra-within (Cholesky):')
        paramRnd2Co_chW_est_vec, paramRnd2Co_chW_se_vec, paramRnd2Co_chW_zVal_vec, paramRnd2Co_chW_pVal_vec, pd_paramRnd2Co_chW = processOutput(est, se, zVal, pVal, lu) 

        print(' ')
        print('Correlated random parameters - intra-within (Cholesky, est.):')
        paramRnd2Co_chW_est = np.zeros((nRnd2Co, nRnd2Co))
        paramRnd2Co_chW_est[ch2Idx] = paramRnd2Co_chW_est_vec
        print(pd.DataFrame(paramRnd2Co_chW_est))
        
        print(' ')
        print('Correlated random parameters - intra-within (Cholesky, std. err.):')
        paramRnd2Co_chW_se = np.zeros((nRnd2Co, nRnd2Co))
        paramRnd2Co_chW_se[ch2Idx] = paramRnd2Co_chW_se_vec
        print(pd.DataFrame(paramRnd2Co_chW_se))
        
        print(' ')
        print('Correlated random parameters - intra-within (Cholesky, p-val.):')
        paramRnd2Co_chW_pVal = np.zeros((nRnd2Co, nRnd2Co))
        paramRnd2Co_chW_pVal[ch2Idx] = paramRnd2Co_chW_pVal_vec  
        print(pd.DataFrame(paramRnd2Co_chW_pVal))
    else:
        paramRnd2Co_mu_est, paramRnd2Co_mu_se, paramRnd2Co_mu_zVal, paramRnd2Co_mu_pVal, pd_paramRnd2Co_mu = None, None, None, None, None
        paramRnd2Co_chB_est, paramRnd2Co_chB_se, paramRnd2Co_chB_pVal, pd_paramRnd2Co_chB = None, None, None, None 
        paramRnd2Co_chW_est, paramRnd2Co_chW_se, paramRnd2Co_chW_pVal, pd_paramRnd2Co_chW = None, None, None, None 
        
    print(' ')
    print('Log-likelihood: ' + str(logLik)) 
    print(' ')
    
    if nRnd:
        print('QMC method: ' + drawsType)
        print('Number of simulation draws: ' 
              + str(nDrawsB) + ' (inter); '
              + str(nDrawsW) + ' (intra)')

    ###
    #Delete draws
    ###      
    
    if deleteDraws:
        drawsUc = None; drawsCo = None; 
        drawsUcB = None; drawsCoB = None; 
        drawsUcW = None; drawsCoW = None; 
    
    ###
    #Save results
    ###
    
    results = {'modelName': modelName, 'seed': seed,
               'estimation_time': toc, 'drawsType': drawsType, 
               'nDraws_inter': nDrawsB,
               'nDraws_intra': nDrawsW,
               'drawsUc': drawsUc, 'drawsCo': drawsCo,
               'logLik': logLik, 'est': est, 'iHess': iHess,
               'paramFix_est': paramFix_est, 'paramFix_se': paramFix_se, 'paramFix_zVal': paramFix_zVal, 
               'paramFix_pVal': paramFix_pVal, 'pd_paramFix': pd_paramFix,
               'paramRndUc_mu_est': paramRndUc_mu_est, 'paramRndUc_mu_se': paramRndUc_mu_se, 
               'paramRndUc_mu_zVal': paramRndUc_mu_zVal,
               'paramRndUc_mu_pVal': paramRndUc_mu_pVal, 'pd_paramRndUc_mu': pd_paramRndUc_mu,
               'paramRndUc_sd_est': paramRndUc_sd_est, 'paramRndUc_sd_se': paramRndUc_sd_se, 
               'paramRndUc_sd_zVal': paramRndUc_sd_zVal,
               'paramRndUc_sd_pVal': paramRndUc_sd_pVal, 'pd_paramRndUc_sd': pd_paramRndUc_sd,
               'paramRndCo_mu_est': paramRndCo_mu_est, 'paramRndCo_mu_se': paramRndCo_mu_se,
               'paramRndCo_mu_zVal': paramRndCo_mu_zVal,
               'paramRndCo_mu_pVal': paramRndCo_mu_pVal, 'pd_paramRndCo_mu': pd_paramRndCo_mu,
               'paramRndCo_ch_est': paramRndCo_ch_est, 'paramRndCo_ch_se': paramRndCo_ch_se,
               'paramRndCo_ch_pVal': paramRndCo_ch_pVal,
               'pd_paramRndCo_ch': pd_paramRndCo_ch,
               'paramRnd2Uc_mu_est': paramRnd2Uc_mu_est, 'paramRnd2Uc_mu_se': paramRnd2Uc_mu_se, 
               'paramRnd2Uc_mu_zVal': paramRnd2Uc_mu_zVal,
               'paramRnd2Uc_mu_pVal': paramRnd2Uc_mu_pVal, 'pd_paramRnd2Uc_mu': pd_paramRnd2Uc_mu,
               'paramRnd2Uc_sdB_est': paramRnd2Uc_sdB_est, 'paramRnd2Uc_sdB_se': paramRnd2Uc_sdB_se, 
               'paramRnd2Uc_sdB_zVal': paramRnd2Uc_sdB_zVal,
               'paramRnd2Uc_sdB_pVal': paramRnd2Uc_sdB_pVal, 'pd_paramRnd2Uc_sdB': pd_paramRnd2Uc_sdB,
               'paramRnd2Uc_sdW_est': paramRnd2Uc_sdW_est, 'paramRnd2Uc_sdW_se': paramRnd2Uc_sdW_se, 
               'paramRnd2Uc_sdW_zVal': paramRnd2Uc_sdW_zVal,
               'paramRnd2Uc_sdW_pVal': paramRnd2Uc_sdW_pVal, 'pd_paramRnd2Uc_sdW': pd_paramRnd2Uc_sdW,
               'paramRnd2Co_mu_est': paramRnd2Co_mu_est, 'paramRnd2Co_mu_se': paramRnd2Co_mu_se,
               'paramRnd2Co_mu_zVal': paramRnd2Co_mu_zVal,
               'paramRnd2Co_mu_pVal': paramRnd2Co_mu_pVal, 'pd_paramRnd2Co_mu': pd_paramRnd2Co_mu,
               'paramRnd2Co_chB_est': paramRnd2Co_chB_est, 'paramRnd2Co_chB_se': paramRnd2Co_chB_se,
               'paramRnd2Co_chB_pVal': paramRnd2Co_chB_pVal,
               'pd_paramRnd2Co_chB': pd_paramRnd2Co_chB,
               'paramRnd2Co_chW_est': paramRnd2Co_chW_est, 'paramRnd2Co_chW_se': paramRnd2Co_chW_se,
               'paramRnd2Co_chW_pVal': paramRnd2Co_chW_pVal,
               'pd_paramRnd2Co_chW': pd_paramRnd2Co_chW,
               'resOpt': resOpt
               }
        
    return results

###
#If main: test
###
    
if __name__ == "__main__":
    
    np.random.seed(4711)

    ###
    #Load data
    ###
    
    data = pd.read_csv('swissmetro_long.csv')
    data = data[((data['PURPOSE'] != 1) & (data['PURPOSE'] != 3)) != True]
    data = data[data['ID'] <= 300]
    
    ###
    #Prepare data
    ###
    
    indID = np.array(data['ID'].values, dtype = 'int64')
    obsID = np.array(data['obsID'].values, dtype = 'int64')
    altID = np.array(data['altID'].values, dtype = 'int64')
    
    chosen = np.array(data['chosen'].values, dtype = 'int64')
    
    tt = np.array(data['TT'].values, dtype = 'float64') / 10
    cost = np.array(data['CO'].values, dtype = 'float64') / 10
    he = np.array(data['HE'].values, dtype = 'float64')/ 10
    ga = np.array(data['GA'].values, dtype = 'int64')
    cost[(altID <= 2) & (ga == 1)] = 0
    
    const2 = 1 * (altID == 2)
    const3 = 1 * (altID == 3)
    """
    ###
    #Generate data
    ###
    
    N = 500
    T = 8
    NT = N * T
    J = 5
    NTJ = NT * J
    
    L = 3 #no. of fixed paramters
    K = 5 #no. of random parameters
    
    true_alpha = np.array([-0.8, 0.8, 1.2])
    true_zeta = np.array([-0.8, 0.8, 1.0, -0.8, 1.5])
    true_Omega = np.array([[1.0, 0.8, 0.8, 0.8, 0.8],
                           [0.8, 1.0, 0.8, 0.8, 0.8],
                           [0.8, 0.8, 1.0, 0.8, 0.8],
                           [0.8, 0.8, 0.8, 1.0, 0.8],
                           [0.8, 0.8, 0.8, 0.8, 1.0]])
    
    xFix = 0 * np.random.rand(NTJ, L)
    xRnd = np.random.rand(NTJ, K)
    xRnd[:,3:] = 0

    betaInd_tmp = true_zeta + \
    (np.linalg.cholesky(true_Omega) @ np.random.randn(K, N)).T
    beta_tmp = np.kron(betaInd_tmp, np.ones((T * J,1)))
    
    eps = -np.log(-np.log(np.random.rand(NTJ,)))
    
    vDet = xFix @ true_alpha + np.sum(xRnd * beta_tmp, axis = 1)
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
    """
    
    ###
    #Estimate MXL via MSLE
    ###
    
    xFix = np.zeros((0,0)) #np.stack((const2, const3), axis = 1)
    xRndUc = np.zeros((0,0)) # #-np.hstack((cost, he, tt))
    xRndCo = np.zeros((0,0))
    
    xRnd2Uc = np.zeros((0,0))
    xRnd2Co = np.stack((cost, he, tt), axis = 1)
    
    #Fixed parameter distributions
    #0: normal
    #1: log-normal (to assure that fixed parameter is striclty negative or positive)
    xFix_trans = np.array([0, 0, 0, 0])
    
    #Random parameter distributions
    #0: normal
    #1: log-normal
    #2: S_B
    xRndUc_trans = np.array([0, 0])
    xRndCo_trans = np.array([0, 0])
    
    xRnd2Uc_trans = np.array([0, 0])
    xRnd2Co_trans = np.array([0, 0])
    
    paramFix_inits = np.zeros((xFix.shape[1],))
    
    paramRndUc_mu_inits = np.zeros((xRndUc.shape[1],))
    paramRndUc_sd_inits = np.ones((xRndUc.shape[1],))
    paramRndCo_mu_inits = np.zeros((xRndCo.shape[1],))
    paramRndCo_ch_inits = 0.1 * np.eye(xRndCo.shape[1])
    
    paramRnd2Uc_mu_inits = np.zeros((xRnd2Uc.shape[1],))
    paramRnd2Uc_sdB_inits = np.ones((xRnd2Uc.shape[1],))
    paramRnd2Uc_sdW_inits = np.ones((xRnd2Uc.shape[1],))
    paramRnd2Co_mu_inits = np.zeros((xRnd2Co.shape[1],))
    paramRnd2Co_chB_inits = 0.1 * np.eye(xRnd2Co.shape[1])
    paramRnd2Co_chW_inits = 0.1 * np.eye(xRnd2Co.shape[1])
    
    drawsType = 'mlhs'
    
    nDrawsB = 200
    nTakesB = 2
    nDrawsW = 200
    nTakesW = 1
    
    seed = 4711

    modelName = 'test'
    deleteDraws = True
    
    results = estimate(
            drawsType, nDrawsB, nTakesB, nDrawsW, nTakesW, 
            seed, modelName, deleteDraws,
            paramFix_inits, 
            paramRndUc_mu_inits, paramRndUc_sd_inits, 
            paramRndCo_mu_inits, paramRndCo_ch_inits,
            paramRnd2Uc_mu_inits, paramRnd2Uc_sdB_inits, paramRnd2Uc_sdW_inits, 
            paramRnd2Co_mu_inits, paramRnd2Co_chB_inits, paramRnd2Co_chW_inits,
            indID, obsID, altID, chosen,
            xFix, xRndUc, xRndCo, xRnd2Uc, xRnd2Co,
            xFix_trans, xRndUc_trans, xRndCo_trans, xRnd2Uc_trans, xRnd2Co_trans)    