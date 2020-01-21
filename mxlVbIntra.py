from joblib import Parallel, delayed
import time
import numpy as np
import scipy as sp
import scipy.sparse
from scipy.stats import invwishart

from mxl import prepareData, pPredMxl
from qmc import makeNormalDraws

###
#E-LSE
###

def calc_vRndB_n(
        paramRndMuB_n, paramRndChB_n,
        xRnd_n, nRnd,
        obsPerInd, nAlt,
        drawsB_n, nDrawsB, nDrawsW_m):    
    paramRndB = paramRndMuB_n.reshape((nRnd,1)) + \
    (paramRndChB_n @ drawsB_n.T)
    vRndB_n = np.sum(xRnd_n.reshape((obsPerInd, nAlt-1, nRnd, 1)) * \
                     paramRndB.reshape((1, 1, nRnd, nDrawsB)), axis = 2)
    vRndB_n = np.repeat(vRndB_n, nDrawsW_m, axis = 2)
    return vRndB_n

def calc_vRndW_n(
        paramRndMuW_n, paramRndChW_n,
        xRnd_n, nRnd,
        obsPerInd, nAlt,
        drawsW_n, nDrawsW):
    paramRndW = paramRndMuW_n.reshape((obsPerInd, nRnd, 1)) + \
    paramRndChW_n @ np.moveaxis(drawsW_n, [1, 2], [2, 1])
    vRndW_n = np.sum(xRnd_n.reshape((obsPerInd, nAlt-1, nRnd, 1)) * \
                     paramRndW.reshape((obsPerInd, 1, nRnd, nDrawsW)), axis = 2)
    return vRndW_n

def calc_vRndW_nt(
        paramRndMuW_nt, paramRndChW_nt,
        xRnd_nt, nRnd,
        nAlt,
        drawsW_nt, nDrawsW):
    paramRndW = paramRndMuW_nt.reshape((nRnd,1)) + \
    paramRndChW_nt @ drawsW_nt.T
    vRndW_nt = np.sum(xRnd_nt.reshape((nAlt-1, nRnd, 1)) * \
                      paramRndW.reshape((1, nRnd, nDrawsW)), axis = 1)
    return vRndW_nt
    
def else_ind(
        paramRndMuB_n, paramRndChB_n, vRndW_n,
        xRnd_n, nRnd, chRndIdx,
        obsPerInd, nAlt,
        drawsB_n, nDrawsB, nDrawsW, nDrawsW_m):
    
    #Utility
    vRndB_n = calc_vRndB_n(
            paramRndMuB_n, paramRndChB_n,
            xRnd_n, nRnd,
            obsPerInd, nAlt,
            drawsB_n, nDrawsB, nDrawsW_m)
    
    v = vRndB_n + vRndW_n
    
    #Probability
    ev = np.exp(v)
    ev[ev > 1e+200] = 1e+200
    ev[ev < 1e-300] = 1e-300
    nev = np.sum(ev, axis = 1, keepdims = True) + 1
    pChosen = np.array(1 / nev).reshape((obsPerInd, nDrawsW))
    pNonChosen = ev / nev
    
    #LSE
    lPChosen = np.log(pChosen)
    lse = np.sum(lPChosen) / nDrawsW
    
    #Gradient
    grRndMu = np.sum(-pNonChosen.reshape((obsPerInd, nAlt-1, 1, nDrawsW)) * \
                     xRnd_n.reshape((obsPerInd, nAlt-1, nRnd, 1)), 
                     axis = (0, 1, 3)) / nDrawsW
    grRndCh = np.sum(-pNonChosen.reshape((obsPerInd, nAlt-1, 1, nDrawsW)) * \
                     xRnd_n.reshape((obsPerInd, nAlt-1, nRnd, 1))[:,:,chRndIdx[0],:] * \
                     np.repeat(drawsB_n.T.reshape((1, 1, nRnd, nDrawsB)), nDrawsW_m, axis = 3)[:,:,chRndIdx[1],:], 
                     axis = (0, 1, 3)) / nDrawsW                  
   
    return lse, grRndMu, grRndCh

def else_obs(
        paramRndMuW_nt, paramRndChW_nt, vRndB_nt,
        xRnd_nt, nRnd, chRndIdx,
        nAlt,
        drawsW_nt, nDrawsW, nDrawsW_m):
    
    #Utility
    vRndW_nt = calc_vRndW_nt(
            paramRndMuW_nt, paramRndChW_nt,
            xRnd_nt, nRnd,
            nAlt,
            drawsW_nt, nDrawsW)
    
    v = vRndB_nt + vRndW_nt
    
    #Probability
    ev = np.exp(v)
    ev[ev > 1e+200] = 1e+200
    ev[ev < 1e-300] = 1e-300
    nev = np.sum(ev, axis = 0, keepdims = True) + 1
    pChosen = np.array(1 / nev).reshape((nDrawsW,))
    pNonChosen = ev / nev
    
    #LSE
    lPChosen = np.log(pChosen)
    lse = np.sum(lPChosen) / nDrawsW
    
    #Gradient
    grRndMu = np.sum(-pNonChosen.reshape((nAlt-1, 1, nDrawsW)) * \
                     xRnd_nt.reshape((nAlt-1, nRnd, 1)), 
                     axis = (0, 2)) / nDrawsW
    grRndCh = np.sum(-pNonChosen.reshape((nAlt-1, 1, nDrawsW)) * \
                     xRnd_nt.reshape((nAlt-1, nRnd, 1))[:,chRndIdx[0],:] * \
                     drawsW_nt.T.reshape((1, nRnd, nDrawsW))[:,chRndIdx[1],:], 
                     axis = (0, 2)) / nDrawsW                  
   
    return lse, grRndMu, grRndCh

###
#Updates
###        
        
def objective_next_paramRndB(
        param, vRndW_n,
        xRnd_n, nRnd, chRndIdx, chRndIdxDiag,
        obsPerInd, nAlt,
        drawsB_n, nDrawsB, nDrawsW, nDrawsW_m,
        zetaMu, psiBInv, omegaB):

    paramRndMuB_n = np.array(param[:nRnd])
    paramRndChB_n = np.zeros((nRnd, nRnd))
    paramRndChB_n[chRndIdx] = np.array(param[nRnd:])
    paramRndChBDiag_n = np.diag(paramRndChB_n)  
    
    ###
    #E-LSE
    ###
    
    lse, grRndMu, grRndCh = else_ind(
            paramRndMuB_n, paramRndChB_n, vRndW_n,
            xRnd_n, nRnd, chRndIdx,
            obsPerInd, nAlt,
            drawsB_n, nDrawsB, nDrawsW, nDrawsW_m)
            
    ###
    #Prior
    ###
    
    lPrior = \
    -(omegaB / 2) * np.trace(paramRndChB_n @ paramRndChB_n.T @ psiBInv) \
    -(omegaB / 2) * paramRndMuB_n @ psiBInv @ paramRndMuB_n \
    + omegaB * paramRndMuB_n @ psiBInv @ zetaMu \
    + np.sum(np.log(np.absolute(paramRndChBDiag_n)))

    grRndMuPrior = -omegaB * psiBInv @ paramRndMuB_n + omegaB * psiBInv @ zetaMu
    grRndChPriorAux = -omegaB * psiBInv @ paramRndChB_n
    grRndChPrior = np.array(grRndChPriorAux[chRndIdx])
    grRndChPrior[chRndIdxDiag] += 1 / paramRndChBDiag_n
    
    ###
    #E-LSE + prior
    ###    
    
    ll = -(lse + lPrior)
    
    grRndMu += grRndMuPrior
    grRndCh += grRndChPrior
    gr = -np.concatenate((grRndMu, grRndCh))
    
    return ll, gr

def objective_next_paramRndW(
        param, vRndB_nt,
        xRnd_nt, nRnd, chRndIdx, chRndIdxDiag,
        nAlt,
        drawsW_nt, nDrawsW, nDrawsW_m,
        psiWInv, omegaW):

    paramRndMuW_nt = np.array(param[:nRnd])
    paramRndChW_nt = np.zeros((nRnd, nRnd))
    paramRndChW_nt[chRndIdx] = np.array(param[nRnd:])
    paramRndChWDiag_nt = np.diag(paramRndChW_nt)  
    
    ###
    #E-LSE
    ###
    
    lse, grRndMu, grRndCh = else_obs(
            paramRndMuW_nt, paramRndChW_nt, vRndB_nt,
            xRnd_nt, nRnd, chRndIdx,
            nAlt,
            drawsW_nt, nDrawsW, nDrawsW_m)
            
    ###
    #Prior
    ###
    
    lPrior = \
    -(omegaW / 2) * np.trace(paramRndChW_nt @ paramRndChW_nt.T @ psiWInv) \
    -(omegaW / 2) * paramRndMuW_nt @ psiWInv @ paramRndMuW_nt \
    + np.sum(np.log(np.absolute(paramRndChWDiag_nt)))

    grRndMuPrior = -omegaW * psiWInv @ paramRndMuW_nt
    grRndChPriorAux = -omegaW * psiWInv @ paramRndChW_nt
    grRndChPrior = np.array(grRndChPriorAux[chRndIdx])
    grRndChPrior[chRndIdxDiag] += 1 / paramRndChWDiag_nt
    
    ###
    #E-LSE + prior
    ###    
    
    ll = -(lse + lPrior)
    
    grRndMu += grRndMuPrior
    grRndCh += grRndChPrior
    gr = -np.concatenate((grRndMu, grRndCh))
    
    return ll, gr

def next_paramRnd(
        local_iter, local_tol,
        paramRndMuB, paramRndChB, paramRndSiB, 
        paramRndMuW, paramRndChW, paramRndSiW,
        xRnd, nRnd, chRndIdx, chRndIdxDiag, intra,
        nInd, obsPerInd, nAlt,
        drawsB, nDrawsB,
        drawsW, nDrawsW, nDrawsW_m,
        zetaMu, psiBInv, omegaB, psiWInv, omegaW): 
        
    for n in np.arange(nInd):
        
        parOld = np.concatenate((paramRndMuB[n,:], 
                                 np.diag(paramRndSiB[n,:,:]),
                                 paramRndMuW[n,:,:].reshape((-1,)),
                                 np.diagonal(paramRndSiW[n,:,:,:], axis1 = 1, axis2 = 2).reshape((-1,))))
        parChange = 1e200
        
        n_iter = 0
        while n_iter < local_iter and parChange >= local_tol:
            n_iter += 1
                   
            ###
            #beta
            ###
            
            xRnd_n = np.array(xRnd[n,:,:,:])
            drawsB_n = np.array(drawsB[n,:,:])
            
            if intra:
                paramRndMuW_n = np.array(paramRndMuW[n,:,:])
                paramRndChW_n = np.array(paramRndChW[n,:,:,:])
                drawsW_n = np.array(drawsW[n,:,:,:])
                vRndW_n = calc_vRndW_n(
                        paramRndMuW_n, paramRndChW_n,
                        xRnd_n, nRnd,
                        obsPerInd, nAlt,
                        drawsW_n, nDrawsW)    
            else:
                vRndW_n = 0
                
            inits = np.concatenate((paramRndMuB[n,:], 
                                    paramRndChB[n,chRndIdx[0],chRndIdx[1]]))
            resOpt = sp.optimize.minimize(
                    fun = objective_next_paramRndB,
                    x0 = inits,
                    args = (vRndW_n,
                            xRnd_n, nRnd, chRndIdx, chRndIdxDiag,
                            obsPerInd, nAlt,
                            drawsB_n, nDrawsB, nDrawsW, nDrawsW_m,
                            zetaMu, psiBInv, omegaB),
                            method = 'L-BFGS-B',
                            jac = True,
                            options = {'disp': False})
            paramRndMuB_n = np.array(resOpt['x'][:nRnd])
            paramRndMuB[n,:] = paramRndMuB_n
            paramRndChB_n = np.zeros((nRnd, nRnd))
            paramRndChB_n[chRndIdx] = np.array(resOpt['x'][nRnd:])
            paramRndChB[n,:,:] = paramRndChB_n
            paramRndSiB[n,:,:] = paramRndChB_n @ paramRndChB_n.T
            
            ###
            #gamma
            ###
            
            if intra:
                vRndB_n = calc_vRndB_n(
                        paramRndMuB_n, paramRndChB_n,
                        xRnd_n, nRnd,
                        obsPerInd, nAlt,
                        drawsB_n, nDrawsB, nDrawsW_m)
                
                for t in np.arange(obsPerInd):
                    vRndB_nt = vRndB_n[t,:,:]
                    
                    xRnd_nt = np.array(xRnd[n,t,:,:])
                    drawsW_nt = np.array(drawsW[n,t,:,:])
                    
                    inits = np.concatenate((paramRndMuW[n,t,:],
                                            paramRndChW[n,t,chRndIdx[0],chRndIdx[1]]))
                    resOpt = sp.optimize.minimize(
                            fun = objective_next_paramRndW,
                            x0 = inits,
                            args = (vRndB_nt,
                                    xRnd_nt, nRnd, chRndIdx, chRndIdxDiag,
                                    nAlt,
                                    drawsW_nt, nDrawsW, nDrawsW_m,
                                    psiWInv, omegaW),
                                    method = 'L-BFGS-B',
                                    jac = True,
                                    options = {'disp': False})
                    paramRndMuW_nt = np.array(resOpt['x'][:nRnd])
                    paramRndMuW[n,t,:] = paramRndMuW_nt
                    paramRndChW_nt = np.zeros((nRnd, nRnd))
                    paramRndChW_nt[chRndIdx] = np.array(resOpt['x'][nRnd:])
                    paramRndChW[n,t,:,:] = paramRndChW_nt
                    paramRndSiW[n,t,:,:] = paramRndChW_nt @ paramRndChW_nt.T
                    
            ###
            #Check for convergence
            ###
            
            parNew = np.concatenate((paramRndMuB[n,:], 
                                 np.diag(paramRndSiB[n,:,:]),
                                 paramRndMuW[n,:,:].reshape((-1,)),
                                 np.diagonal(paramRndSiW[n,:,:,:], axis1 = 1, axis2 = 2).reshape((-1,))))
            parChange = np.max(np.absolute((parNew - parOld)) / 
                               np.absolute(parOld + 1e-8))
            parOld = parNew
    
    return (paramRndMuB, paramRndChB, paramRndSiB,
            paramRndMuW, paramRndChW, paramRndSiW)
    
###
#Batch update
###
    
def batchUpdate_svb(
        local_iter, local_tol,
        paramRndMuB_k, paramRndChB_k, paramRndSiB_k,
        paramRndMuW_k, paramRndChW_k, paramRndSiW_k,
        zetaMu, zetaSi,
        psiB, psiBInv, omegaB, dK_B,
        psiW, psiWInv, omegaW, dK_W,
        nu, cK, rK,
        mu0Rnd, Sigma0RndInv,
        xRnd, nRnd, chRndIdx, chRndIdxDiag, intra,
        nInd, obsPerInd, nAlt, N_K, T_K,
        drawsB, nDrawsB,
        drawsW, nDrawsW, nDrawsW_m):
    
    ###
    #Local updates
    ###
    
    #beta and gamma
    (paramRndMuB_k, paramRndChB_k, paramRndSiB_k,
     paramRndMuW_k, paramRndChW_k, paramRndSiW_k) = next_paramRnd(
            local_iter, local_tol,
            paramRndMuB_k, paramRndChB_k, paramRndSiB_k, 
            paramRndMuW_k, paramRndChW_k, paramRndSiW_k,
            xRnd, nRnd, chRndIdx, chRndIdxDiag, intra,
            N_K, obsPerInd, nAlt,
            drawsB, nDrawsB,
            drawsW, nDrawsW, nDrawsW_m,
            zetaMu, psiBInv, omegaB, psiWInv, omegaW)

    ###
    #Intermediate global updates
    ###
    
    weightInd = nInd / N_K
    weightObs = weightInd * (obsPerInd / T_K)
    
    #zeta
    zetaSiInv_k = Sigma0RndInv + nInd * omegaB * psiBInv
    zetaSi_k = np.linalg.inv(zetaSiInv_k)
    zetaMu_k = np.linalg.solve(zetaSiInv_k,
                               Sigma0RndInv @ mu0Rnd +\
                               weightInd * omegaB * psiBInv @ np.sum(paramRndMuB_k, axis = 0))

    #Omega_B
    xsB = paramRndMuB_k - zetaMu_k
    psiB_k = 2 * nu * np.diag(cK / dK_B) + nInd * zetaSi_k +\
    weightInd * (np.sum(paramRndSiB_k, axis = 0) + xsB.T @ xsB)

    #Omega_W
    if intra:
        psiW_k = 2 * nu * np.diag(cK / dK_W) +\
        weightObs * (np.sum(paramRndSiW_k, axis = (0, 1)) + \
        paramRndMuW_k.reshape((N_K * obsPerInd, nRnd)).T @ \
        paramRndMuW_k.reshape((N_K * obsPerInd, nRnd)))
    else:
        psiW_k = psiW
        
    return zetaMu_k, zetaSiInv_k, psiB_k, psiW_k

def batchUpdate_vb(
        local_iter, local_tol,
        paramRndMuB_k, paramRndChB_k, paramRndSiB_k,
        paramRndMuW_k, paramRndChW_k, paramRndSiW_k,
        zetaMu,
        psiBInv, omegaB,
        psiWInv, omegaW,
        xRnd, nRnd, chRndIdx, chRndIdxDiag, intra,
        obsPerInd, nAlt, N_K,
        drawsB, nDrawsB,
        drawsW, nDrawsW, nDrawsW_m):
    
    ###
    #Local updates
    ###
    
    #beta and gamma
    (paramRndMuB_k, paramRndChB_k, paramRndSiB_k,
     paramRndMuW_k, paramRndChW_k, paramRndSiW_k) = next_paramRnd(
            local_iter, local_tol,
            paramRndMuB_k, paramRndChB_k, paramRndSiB_k, 
            paramRndMuW_k, paramRndChW_k, paramRndSiW_k,
            xRnd, nRnd, chRndIdx, chRndIdxDiag, intra,
            N_K, obsPerInd, nAlt,
            drawsB, nDrawsB,
            drawsW, nDrawsW, nDrawsW_m,
            zetaMu, psiBInv, omegaB, psiWInv, omegaW)
        
    return (paramRndMuB_k, paramRndChB_k, paramRndSiB_k,
            paramRndMuW_k, paramRndChW_k, paramRndSiW_k)

###
#Coordinate ascent
###
    
def sg_update(l_new, l_old, rho):
    g = l_new - l_old
    l = l_old + rho * g
    return l
    
def coordinateAscent_svb(
        vb_iter, vb_tol,
        svb_eta, svb_kappa,
        local_iter, local_tol,
        zetaMu, zetaSi,
        psiB, psiBInv, omegaB, dK_B,
        psiW, psiWInv, omegaW, dK_W,
        nu, cK, rK,
        mu0Rnd, Sigma0RndInv,
        xRnd, nRnd, chRndIdx, chRndIdxDiag, intra,
        nInd, obsPerInd, nAlt,
        K, N_K, T_K,
        drawsB, nDrawsB,
        drawsW, nDrawsW, nDrawsW_m):
    
    ###
    #Initialise
    ###
    
    zetaSiInv = np.linalg.inv(zetaSi)
    zetaSiInvMu = np.linalg.solve(zetaSi, zetaMu.reshape((nRnd,1)))
    psiB_l = -0.5 * psiB
    psiW_l = -0.5 * psiW
    
    paramRndMuB_K = [np.zeros((N_K, nRnd))] * K
    paramRndSiB_K = [np.repeat(0.01 * np.eye(nRnd).reshape(1, nRnd, nRnd), N_K, axis = 0)] * K
    paramRndChB_K = [np.linalg.cholesky(i) for i in paramRndSiB_K]
    paramRndMuW_K = [np.zeros((N_K, obsPerInd, nRnd))] * K
    paramRndSiW_K = [np.repeat(0.01 * np.eye(nRnd).reshape(1, nRnd, nRnd),\
                               N_K * obsPerInd, axis = 0).reshape((N_K, obsPerInd, nRnd, nRnd))] * K
    paramRndChW_K = [np.linalg.cholesky(i) for i in paramRndSiW_K]      
    
    iters = 0
    parOld = np.concatenate((zetaMu, np.diag(psiB), dK_B, np.diag(psiW), dK_W))
    parmat = np.zeros((5, parOld.shape[0]))
    parChange = 1e200
    parChangeOld = parChange
    parChangeDiff = 0
    
    ###
    #CAVI
    ###
    
    while iters < vb_iter and parChange >= vb_tol and parChangeDiff < 0.1:
        iters += 1
        
        ###
        #Learning rate
        ###
        
        alpha = min(svb_kappa, iters) / svb_kappa
        rho = (1 - alpha) * svb_eta + alpha * svb_eta * 0.01   
        
        ###
        #Make batches 
        ###
        
        inds = np.random.choice(nInd, size = K * N_K, replace = False).reshape((K, N_K))
        xRnd_K = []
        for k in np.arange(K):
            xRnd_k = np.zeros((N_K, T_K, nAlt-1, nRnd))
            for n in np.arange(N_K):
                obs = np.random.choice(obsPerInd, size = T_K, replace = False)
                xRnd_k[n,:,:,:] = xRnd[inds[k,n],obs,:,:]
            xRnd_K.append(xRnd_k)        
        
        ###
        #Batch updates
        ###
    
        zetaMu_K = np.zeros((K, nRnd))
        zetaSiInv_K = np.zeros((K, nRnd, nRnd))
        psiB_K = np.zeros((K, nRnd, nRnd))
        psiW_K = np.zeros((K, nRnd, nRnd))
        
        aux = Parallel(n_jobs = K)(delayed(batchUpdate_svb)(
                local_iter, local_tol,
                paramRndMuB_K[k], paramRndChB_K[k], paramRndSiB_K[k],
                paramRndMuW_K[k], paramRndChW_K[k], paramRndSiW_K[k],
                zetaMu, zetaSi,
                psiB, psiBInv, omegaB, dK_B,
                psiW, psiWInv, omegaW, dK_W,
                nu, cK, rK,
                mu0Rnd, Sigma0RndInv,
                xRnd_K[k], nRnd, chRndIdx, chRndIdxDiag, intra,
                nInd, obsPerInd, nAlt, N_K, T_K,
                drawsB, nDrawsB,
                drawsW, nDrawsW, nDrawsW_m)
            for k in range(K))
        
        for k in np.arange(K):
            zetaMu_K[k,:], zetaSiInv_K[k,:,:], psiB_K[k,:,:], psiW_K[k,:,:] = aux[k]
        
        ###
        #Global updates
        ###
        
        #zeta
        zetaSiInv_new = np.mean(zetaSiInv_K, axis = 0)
        zetaSiInv = sg_update(zetaSiInv_new, zetaSiInv, rho)
        zetaSi = np.linalg.inv(zetaSiInv)
        
        zetaSiInvMu_new = np.mean(zetaSiInv_K @ zetaMu_K.reshape((K, nRnd, 1)), axis = 0)
        zetaSiInvMu = sg_update(zetaSiInvMu_new, zetaSiInvMu, rho)
        zetaMu = np.linalg.solve(zetaSiInv, zetaSiInvMu).reshape((nRnd, ))

        #Omega_B
        psiB_l_new = -0.5 * np.mean(psiB_K, axis = 0)
        psiB_l = sg_update(psiB_l_new, psiB_l, rho)  
        psiB = -2 * psiB_l
        psiBInv = np.linalg.inv(psiB)
        
        #Omega_W
        if intra:
            psiW_l_new = -0.5 * np.mean(psiW_K, axis = 0)
            psiW_l = sg_update(psiW_l_new, psiW_l, rho)        
            psiW = -2 * psiW_l
            psiWInv = np.linalg.inv(psiW)
        
        #iwishDiagA
        dK_B = rK + nu * omegaB * np.diag(psiBInv)
        if intra:
            dK_W = rK + nu * omegaW * np.diag(psiWInv)
        
        ###
        #Check for convergence
        ###
        
        par = np.concatenate((zetaMu, np.diag(psiB), dK_B, np.diag(psiW), dK_W)).reshape((1,-1))
        parmat = np.vstack((parmat[1:,:], par))
        parNew = np.mean(parmat, axis = 0)
        if iters > 5:   
            parChange = np.max(np.absolute((parNew - parOld)) / 
                               np.absolute(parOld + 1e-8))
            parChangeDiff = parChange - parChangeOld
            parChangeOld = parChange
        parOld = parNew
          
        ###
        #Display progress
        ###
        
        print(" ")
        print('Iteration ' + str(iters) + ' (SVB)' +
              '; max. rel. change param.: ' + str(parChange) + ';')
        print("Learning rate: " + str(rho))
        
        print("zetaMu:")
        print(zetaMu)
        print("diag(psiB):")
        print(np.diag(psiB))
        print("dK_B:")
        print(dK_B) 
        if intra:
            print("diag(psiW):")
            print(np.diag(psiW))
            print("dK_W:")
            print(dK_W) 

    return (zetaMu, zetaSi, 
            psiB, dK_B, 
            psiW, dK_W,
            iters, parChange)

def coordinateAscent_vb(
        vb_iter, vb_tol,
        local_iter, local_tol,
        paramRndMuB, paramRndChB, paramRndSiB,
        paramRndMuW, paramRndChW, paramRndSiW,
        zetaMu, zetaSi,
        psiB, psiBInv, omegaB, dK_B,
        psiW, psiWInv, omegaW, dK_W,
        nu, cK, rK,
        mu0Rnd, Sigma0RndInv,
        xRnd, nRnd, chRndIdx, chRndIdxDiag, intra,
        nInd, obsPerInd, nAlt,
        K, N_K,
        drawsB, nDrawsB,
        drawsW, nDrawsW, nDrawsW_m):
    
    ###
    #Initialise
    ###
    
    iters = 0
    parOld = np.concatenate((zetaMu, np.diag(psiB), dK_B, np.diag(psiW), dK_W))
    parmat = np.zeros((5, parOld.shape[0]))
    parChange = 1e200
    parChangeOld = parChange
    parChangeDiff = 0
    
    ###
    #CAVI
    ###
    
    #Make batches
    inds = np.arange(nInd).reshape((K, N_K))
    xRnd_K = [np.array(xRnd[inds[k,:],:,:,:]) for k in np.arange(K)]     
    
    while iters < vb_iter and parChange >= vb_tol and parChangeDiff < 0.1:
        iters += 1
        
        #beta and gamma        
        aux = Parallel(n_jobs = K)(delayed(batchUpdate_vb)(
                local_iter, local_tol,
                np.array(paramRndMuB[inds[k,:],:]), 
                np.array(paramRndChB[inds[k,:],:,:]), 
                np.array(paramRndSiB[inds[k,:],:,:]),
                np.array(paramRndMuW[inds[k,:],:,:]), 
                np.array(paramRndChW[inds[k,:],:,:,:]), 
                np.array(paramRndSiW[inds[k,:],:,:,:]),
                zetaMu,
                psiBInv, omegaB,
                psiWInv, omegaW,
                xRnd_K[k], nRnd, chRndIdx, chRndIdxDiag, intra,
                obsPerInd, nAlt, N_K,
                np.array(drawsB[inds[k,:],:]), nDrawsB,
                np.array(drawsW[inds[k,:],:,:]), nDrawsW, nDrawsW_m)
            for k in range(K))
        
        for k in np.arange(K):
            paramRndMuB[inds[k,:],:] = aux[k][0]
            paramRndChB[inds[k,:],:,:] = aux[k][1]
            paramRndSiB[inds[k,:],:,:] = aux[k][2]
            paramRndMuW[inds[k,:],:,:] = aux[k][3] 
            paramRndChW[inds[k,:],:,:,:] = aux[k][4] 
            paramRndSiW[inds[k,:],:,:,:] = aux[k][5]

        #zeta
        zetaSiInv = Sigma0RndInv + nInd * omegaB * psiBInv
        zetaSi = np.linalg.inv(zetaSiInv)
        zetaMu = np.linalg.solve(zetaSiInv,
                                 Sigma0RndInv @ mu0Rnd +\
                                 omegaB * psiBInv @ np.sum(paramRndMuB, axis = 0))

        #Omega_B
        xsB = paramRndMuB - zetaMu
        psiB = 2 * nu * np.diag(cK / dK_B) + nInd * zetaSi +\
        np.sum(paramRndSiB, axis = 0) + xsB.T @ xsB
        psiBInv = np.linalg.inv(psiB)
        
        #Omega_W
        if intra:
            psiW = 2 * nu * np.diag(cK / dK_W) +\
            np.sum(paramRndSiW, axis = (0, 1)) + \
            paramRndMuW.reshape((nInd * obsPerInd, nRnd)).T @ paramRndMuW.reshape((nInd * obsPerInd, nRnd))
            psiWInv = np.linalg.inv(psiW)
        
        #iwishDiagA
        dK_B = rK + nu * omegaB * np.diag(psiBInv)
        if intra:
            dK_W = rK + nu * omegaW * np.diag(psiWInv)
        
        ###
        #Check for convergence
        ###
        
        par = np.concatenate((zetaMu, np.diag(psiB), dK_B, np.diag(psiW), dK_W)).reshape((1,-1))
        parmat = np.vstack((parmat[1:,:], par))
        parNew = np.mean(parmat, axis = 0)
        if iters > 5:   
            parChange = np.max(np.absolute((parNew - parOld)) / 
                               np.absolute(parOld + 1e-8))
            parChangeDiff = parChange - parChangeOld
            parChangeOld = parChange
        parOld = parNew
          
        ###
        #Display progress
        ###
        
        print(" ")
        print('Iteration ' + str(iters) + ' (VB)' +
              '; max. rel. change param.: ' + str(parChange) + ';')
        
        print("zetaMu:")
        print(zetaMu)
        print("diag(psiB):")
        print(np.diag(psiB))
        print("dK_B:")
        print(dK_B) 
        if intra:
            print("diag(psiW):")
            print(np.diag(psiW))
            print("dK_W:")
            print(dK_W) 
        print("paramRndMuB:")
        print(paramRndMuB[:10,:])

    return (paramRndMuW, paramRndSiW,
            paramRndMuB, paramRndSiB,
            zetaMu, zetaSi, 
            psiB, dK_B, 
            psiW, dK_W,
            iters, parChange)
    
###
#Prepare data
###
    
def prepareData_vb(xRnd, indID, obsID, chosen):
    nInd = np.unique(indID).shape[0]
    nObs = np.unique(obsID).shape[0] 
    nRowFull = indID.shape[0]
    obsPerInd = np.unique(np.unique(np.stack((indID, obsID), axis = 1),
                                    axis = 0)[:,0], return_counts = True)[1][0]
    nAlt = np.unique(obsID, return_counts = True)[1][0]
    nRnd = xRnd.shape[1]
    
    chosenIdx = (np.arange(nRowFull) + 1) * chosen
    chosenIdx = chosenIdx[chosenIdx > 0] - 1
    nonChosenIdx = (np.arange(nRowFull) + 1) * (chosen == 0)
    nonChosenIdx = nonChosenIdx[nonChosenIdx > 0] - 1
    
    b = 0
    for i in np.arange(nObs):
        a = b; b += nAlt; idx = slice(a, b);
        xRnd[idx, :] = xRnd[idx, :] - xRnd[chosenIdx[i], :] 
    xRnd = np.array(xRnd[nonChosenIdx, :])
    xRnd = xRnd.reshape((nInd, obsPerInd, nAlt-1, -1))
    
    return (xRnd,
            nInd, nObs, obsPerInd, nAlt, nRnd,
            chosenIdx, nonChosenIdx)

###
#Estimate
###

def estimate(
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
        xRnd, intra):
    
    assert svb or vb, "Method not supported!"
    
    np.random.seed(seed)
    
    if intra:
        local_iter = 1
    
    ###
    #Prepare data
    ###
    
    (xRnd,
     nInd, nObs, obsPerInd, nAlt, nRnd,
     chosenIdx, nonChosenIdx) = prepareData_vb(xRnd, indID, obsID, chosen)
    
    chRndIdx = np.triu_indices(nRnd); chRndIdx = chRndIdx[1], chRndIdx[0];
    chRndIdxDiagAux = np.ones((nRnd, nRnd), dtype = 'int64')
    chRndIdxDiagAux[chRndIdx] = np.arange((nRnd * (nRnd + 1) / 2))
    chRndIdxDiag = np.diag(chRndIdxDiagAux)
    
    tic = time.time()
    
    ###
    #SVB/VB
    ###
    
    #Initialise
    zetaMu = zetaMu_inits.copy()
    zetaSi = zetaSi_inits.copy()
    psiB = psiB_inits.copy()
    psiBInv = np.linalg.inv(psiB)
    dK_B = dK_B_inits.copy()
    psiW = psiW_inits.copy()
    psiWInv = np.linalg.inv(psiW)
    dK_W = dK_W_inits.copy()
    
    if svb:
        
        ### 
        #Generate draws
        ###
        
        if not intra: 
            nDrawsW_m = 1
        nDrawsW = nDrawsB * nDrawsW_m
        
        drawsB = makeNormalDraws(nDrawsB, nRnd, drawsType, N_K)[0]
        if intra:
            drawsW = makeNormalDraws(nDrawsW, nRnd, drawsType, N_K * T_K)[0]
            drawsW = drawsW.reshape((N_K, T_K, nDrawsW, nRnd))
        else:
            drawsW = np.zeros((N_K, T_K, nDrawsW, nRnd))
        
        ### 
        #Coordinate Ascent
        ###
    
        (zetaMu, zetaSi, 
         psiB, dK_B, 
         psiW, dK_W,
         iters, parChange) = coordinateAscent_svb(
                 vb_iter, vb_tol,
                 svb_eta, svb_kappa,
                 local_iter, local_tol,
                 zetaMu, zetaSi,
                 psiB, psiBInv, omegaB, dK_B,
                 psiW, psiWInv, omegaW, dK_W,
                 nu, cK, rK,
                 mu0Rnd, Sigma0RndInv,
                 xRnd, nRnd, chRndIdx, chRndIdxDiag, intra,
                 nInd, obsPerInd, nAlt,
                 K[0], N_K, T_K,
                 drawsB, nDrawsB,
                 drawsW, nDrawsW, nDrawsW_m)    
    
    if vb:   
        ### 
        #Generate draws
        ###
        
        if not intra: 
            nDrawsW_m = 1
        nDrawsW = nDrawsB * nDrawsW_m
        
        drawsB = makeNormalDraws(nDrawsB, nRnd, drawsType, nInd)[0]
        if intra:
            drawsW = makeNormalDraws(nDrawsW, nRnd, drawsType, nObs)[0]
            drawsW = drawsW.reshape((nInd, obsPerInd, nDrawsW, nRnd))
        else:
            drawsW = np.zeros((nInd, obsPerInd, nDrawsW, nRnd))
        
        ### 
        #Coordinate Ascent
        ###
        
        paramRndMuB = paramRndMuB_inits.copy()
        paramRndSiB = paramRndSiB_inits.copy()
        paramRndChB = np.linalg.cholesky(paramRndSiB)
        paramRndMuW = paramRndMuW_inits.copy().reshape((nInd, obsPerInd, nRnd))
        paramRndSiW = paramRndSiW_inits.copy().reshape((nInd, obsPerInd, nRnd, nRnd))
        paramRndChW = np.linalg.cholesky(paramRndSiW)
    
        (paramRndMuW, paramRndSiW,
         paramRndMuB, paramRndSiB,
         zetaMu, zetaSi, 
         psiB, dK_B, 
         psiW, dK_W,
         iters, parChange) = coordinateAscent_vb(
                 vb_iter, vb_tol,
                 local_iter, local_tol,
                 paramRndMuB, paramRndChB, paramRndSiB,
                 paramRndMuW, paramRndChW, paramRndSiW,
                 zetaMu, zetaSi,
                 psiB, psiBInv, omegaB, dK_B,
                 psiW, psiWInv, omegaW, dK_W,
                 nu, cK, rK,
                 mu0Rnd, Sigma0RndInv,
                 xRnd, nRnd, chRndIdx, chRndIdxDiag, intra,
                 nInd, obsPerInd, nAlt,
                 K[1], int(nInd / K[1]),
                 drawsB, nDrawsB,
                 drawsW, nDrawsW, nDrawsW_m)
    
    ###
    #Estimation time
    ###    
    
    toc = time.time() - tic
    
    print(' ')
    print('Computation time [s]: ' + str(toc))
    
    ###
    #Save results
    ###
    
    results = {'modelName': modelName,
               'estimation_time': toc, 'iters': iters, 'termTol': parChange,
               'paramRndMuB': paramRndMuB, 'paramRndSiB': paramRndSiB,
               'paramRndMuW': paramRndMuW, 'paramRndSiW': paramRndSiW,
               'zetaMu': zetaMu, 'zetaSi': zetaSi, 
               'psiB': psiB, 'dK_B': dK_B, 'omegaB': omegaB,
               'psiW': psiW, 'dK_W': dK_W, 'omegaW': omegaW,
               }
        
    return results

def inits(indID, obsID, xRnd, nu, A):
    nRnd = xRnd.shape[1]
    nInd = np.unique(indID).shape[0]
    nObs = np.unique(obsID).shape[0]
    
    paramRndMuB_inits = np.zeros((nInd, nRnd))
    paramRndSiB_inits = np.repeat(0.01 * np.eye(nRnd).reshape(1, nRnd, nRnd),
                                  nInd, axis = 0)
    paramRndMuW_inits = np.zeros((nObs, nRnd))
    paramRndSiW_inits = np.repeat(0.01 * np.eye(nRnd).reshape(1, nRnd, nRnd),
                                  nObs, axis = 0)
    zetaMu_inits = np.zeros((nRnd,))
    zetaSi_inits = 0.01 * np.eye(nRnd)
    cK = (nu + nRnd) / 2.0
    rK = A**(-2) * np.ones((nRnd,))
    dK_B_inits = cK * np.ones((nRnd,))
    dK_W_inits = dK_B_inits.copy()
    omegaB = nu + nInd + nRnd - 1
    omegaW = nu + nObs + nRnd - 1
    psiB_inits = (omegaB - nRnd + 1) * np.eye(nRnd)
    psiW_inits = (omegaW - nRnd + 1) * np.eye(nRnd) 
    
    return (paramRndMuB_inits, paramRndSiB_inits,
            paramRndMuW_inits, paramRndSiW_inits,
            zetaMu_inits, zetaSi_inits,
            cK, rK,
            omegaB, psiB_inits, dK_B_inits,
            omegaW, psiW_inits, dK_W_inits)

###
#Prediction
###
    

def predictB(
        nIter, nTakes, nSim, seed,
        zetaMu, zetaSi, psiB, omegaB, psiW, omegaW,
        indID, obsID, altID, chosen,
        xRnd):
    
    np.random.seed(seed)
    
    ###
    #Prepare data
    ###
    
    nRnd = xRnd.shape[1]
    
    xList = [xRnd]
    (xList,
     nInd, nObs, nRow,
     chosenIdx, nonChosenIdx,
     rowsPerInd, rowsPerObs,
     _, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen)
    xRnd = xList[0]
    
    sim_xRnd = np.tile(xRnd, (nSim, 1))
    sim_rowsPerObs = np.tile(rowsPerObs, (nSim,))
    sim_map_avail_to_obs = scipy.sparse.kron(scipy.sparse.eye(nSim), map_avail_to_obs)

    ###
    #Prediction
    ###
    
    pPred = np.zeros((nRow + nObs,))  
    vFix = 0 
    
    zetaCh = np.linalg.cholesky(zetaSi)
    
    for i in np.arange(nIter):
        zeta_tmp = zetaMu + zetaCh @ np.random.randn(nRnd,)
        chB_tmp = np.linalg.cholesky(invwishart.rvs(omegaB, psiB).reshape((nRnd, nRnd)))
        chW_tmp = np.linalg.cholesky(invwishart.rvs(omegaW, psiW).reshape((nRnd, nRnd)))
        
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
        
    pPred /= nIter  
    return pPred
    
def predictW(
        nIter, nTakes, nSim, seed,
        paramRndMuB, paramRndSiB, psiW, omegaW,
        indID, obsID, altID, chosen,
        xRnd):
    
    np.random.seed(seed)
    
    ###
    #Prepare data
    ###
    
    nRnd = xRnd.shape[1]
    
    xList = [xRnd]
    (xList,
     nInd, nObs, nRow,
     chosenIdx, nonChosenIdx,
     rowsPerInd, rowsPerObs,
     _, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen)
    xRnd = xList[0]
    
    sim_xRnd = np.tile(xRnd, (nSim, 1))
    sim_rowsPerObs = np.tile(rowsPerObs, (nSim,))
    sim_map_avail_to_obs = scipy.sparse.kron(scipy.sparse.eye(nSim), map_avail_to_obs)

    ###
    #Prediction
    ###
    
    pPred = np.zeros((nRow + nObs,))  
    vFix = 0
    
    paramRndChB = np.linalg.cholesky(paramRndSiB)
    
    for i in np.arange(nIter):
        paramRndB_tmp = np.tile(paramRndMuB + \
                                (paramRndChB @ np.random.randn(nInd, nRnd, 1)).reshape((nInd, nRnd,)), 
                                (nSim,1))
        chW_tmp = np.linalg.cholesky(invwishart.rvs(omegaW, psiW).reshape((nRnd, nRnd)))
        
        pPred_iter = np.zeros((nRow + nObs,))
        
        for t in np.arange(nTakes):
            paramRnd = paramRndB_tmp + (chW_tmp @ np.random.randn(nRnd, nObs * nSim)).T
            paramRndPerRow = np.repeat(paramRnd, sim_rowsPerObs, axis = 0)
            vRnd = np.sum(sim_xRnd * paramRndPerRow, axis = 1)
            
            pPred_take = pPredMxl(vFix, vRnd, sim_map_avail_to_obs, nSim, chosenIdx, nonChosenIdx)
            pPred_iter += pPred_take
            
        pPred += pPred_iter / nTakes
    pPred /= nIter  
    return pPred