#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:42:08 2020

@author: amishra
"""


# load required python module 
import random    
import pandas as pd
import numpy as np
from scipy.stats import norm
import pystan
import argparse
import pickle
import sys
import os 
utils_dir= os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(utils_dir)
import sub_fun as sf
import vb_stan as vbfun
# -------------------------
#  Command-Line Argument for Path Selection
# -------------------------

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


data_dir = os.path.join(base_dir, "data/data_new/")
stan_mod= os.path.join(base_dir, "stan_model/")
#stan_mod = os.path.join(base_dir, "stan_model/NB_microbe_ppc_test.stan")
results_dir = os.path.join(base_dir, "results/results_new_var/component/")

diag_dir = os.path.join(results_dir, "diagnostics/")
model_dir = os.path.join(results_dir, "models/")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(diag_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Get setting parameter for running the script
print(sys.argv)
[l,m_seed,sp_mean,sp_var, h_prop, uid, mtype] = map(float, sys.argv[1:])
uid = int(uid); mtype = int(mtype); m_seed = int(m_seed); l = int(l); uid = int(uid); 
m_seed = int(m_seed); l = int(l)


'''
# lanent_rank [l]; model seed [m_seed] 
# regularization of the mean parameter [sp_mean]
# regularization of the dispersion parameter [sp_var]
# holdout proporion of the test sample [h_prop]
# number of posterior samples from the variational posterior distribution [mtype]
# identifier for the simulation seting: uid
# identifier for the selected seting: sid
'''

## local test setting  
# l = 2; m_seed = 123;  sp_mean = 10;  sp_var = 1; 
# h_prop = 0.1;mtype = 100; uid = 123; sid = 2

'''
Import data for model fitting
'''

## Response matrix: microbial abundance data 
y_path= os.path.join(data_dir, "Y1.csv") #change the path of the data
x_path = os.path.join(data_dir, "X.csv")
z_path = os.path.join(data_dir, "Z.csv")
d_path = os.path.join(data_dir, "satellite.csv")

#Y = pd.read_csv(y_path).to_numpy()  
#Y = Y[:,range(2,Y.shape[1])]
#Y = Y.astype('int')
## Response matrix: microbial abundance data 
Y = pd.read_csv(y_path).to_numpy()  
Y = Y[:,range(2,Y.shape[1])]
Y = Y.astype('int')

## Computation of the geometric mean:  
errx = 1e-5
delta  = np.empty(Y.shape[0])  
for i in range(Y.shape[0]):
    delta[i] = sf.get_geomean(Y[i], errx)
    
T = np.exp(np.mean(np.log(Y+delta.min()), axis=1))
Bs = np.sum(Y != 0, axis = 1)
Yi = (Y != 0) + 0

# Correction for the geometric mean 
T_i = np.exp(np.mean(np.log(Y.T+delta), axis=0))
Y = (Y.T+delta).T
Y = Y.astype('int')

## Geochemical covariates 
X = pd.read_csv(x_path).iloc[:,1:].to_numpy()    
X = np.subtract(X, np.mean(X, axis = 0)) # mean centering
X = X/np.std(X,axis=0)                   # scaling 

#new variables 
D = pd.read_csv(d_path).iloc[:,1:].to_numpy()    
D = np.subtract(D, np.mean(D, axis = 0)) # mean centering
D = D/np.std(D,axis=0) #scaling 

## Spatio-temporal indicators
Z = pd.read_csv(z_path)
I = Z.to_numpy()[:,range(1,Z.shape[1])]   
     
# B biome indicator 
Ifac = I[:,0]
fac = np.unique(Ifac)
B = np.zeros((X.shape[0], fac.shape[0]))
for i in range(fac.shape[0]):
    B[np.where(Ifac == fac[i]),i] = 1
    
# Longhurst province indicator for spatial location
Ifac = I[:,1]
fac = np.unique(Ifac)
S = np.zeros((X.shape[0], fac.shape[0]))
for i in range(fac.shape[0]):
    S[np.where(Ifac == fac[i]),i] = 1
    

# Q quarter indicator for time;
Ifac = I[:,4]
fac = np.unique(Ifac)
Q = np.zeros((X.shape[0], fac.shape[0]))
for i in range(fac.shape[0]):
    Q[np.where(Ifac == fac[i]),i] = 1
    
    
    
'''
Full data analysis, model diagnostic and posterior predictive check for model comparison
'''

# construct 'holdout_mask': an indicator matrix for training and testing data 
n, q = Y.shape
holdout_portion = h_prop
# Ensure `n_holdout` does not exceed `n * q`
# Ensure `n_holdout` does not exceed `n * q`
n_holdout = int(holdout_portion * n * q)

if n_holdout > (n * q):  # Prevents crash
    print(f"Warning: Reducing n_holdout from {n_holdout} to {n * q} (max available)")
    n_holdout = n * q

#n_holdout = int(holdout_portion * n * q)
holdout_mask  = np.zeros(n*q)
random.seed(m_seed)
if (holdout_portion > 0.):
    tem  = np.random.choice(range(n * q), size = n_holdout, replace = False)
    holdout_mask[tem] = 1. 
holdout_mask = holdout_mask.reshape((n,q))


# training and validation set for the analysis 
Y_train = np.multiply(1-holdout_mask, Y)
Y_vad = np.multiply(holdout_mask, Y)



'''
Prepare input data, compile stan model and define output file (to store the model output)
'''

data = {'n':Y.shape[0],'q':Y.shape[1],'p':X.shape[1],'l': l,'s':S.shape[1], "d": D.shape[1], \
        'b':B.shape[1], 'Y':Y, 'X':X, 'S':S, 'B':B, 'Yi':Yi, 'T':T_i, 'Bs':Bs, "D": D, \
        'holdout': holdout_mask, 'sp_mean' : sp_mean, 'sp_var' : sp_var,\
        'm':Q.shape[1], 'Q': Q}

if mtype == 0:
    fname = os.path.join(stan_mod, 'NB_microbe_ppc.stan') ## stan model file name
    tol_rel_obj_set = 0.01        # convergence criteria vb
if mtype == 1:
    fname = os.path.join(stan_mod,'NB_microbe_ppc_nointer.stan')
    tol_rel_obj_set = 0.01
if mtype == 2:
    fname = os.path.join(stan_mod, 'NB_microbe_ppc-G.stan')
    tol_rel_obj_set = 0.01
if mtype == 3:
    fname = os.path.join(stan_mod, 'NB_microbe_ppc-1.stan')
    tol_rel_obj_set = 0.01
if mtype == 4:
    fname = os.path.join(stan_mod, 'NB_microbe_ppc-2.stan')
    tol_rel_obj_set = 0.01
if mtype == 5:
    fname = os.path.join(stan_mod, 'NB_microbe_ppc-3.stan')
    tol_rel_obj_set = 0.01
if mtype == 6:
    fname = os.path.join(stan_mod, "NB_microbe_ppc_test_new.stan")
    tol_rel_obj_set = 0.01        


model_NB = open(fname, 'r').read()          # read model file 
mod = pystan.StanModel(model_code=model_NB) # model compile 

# model output file 
sample_file_o = os.path.join(diag_dir, f"{uid}_{mtype}_{m_seed}_nb_sample.csv")    ## posterior sample file 
diag_file_o = os.path.join(diag_dir, f"{uid}_{mtype}_{m_seed}_nb_diag.csv")        ## variational bayes model diagnostic file 
model_output_file = os.path.join(model_dir, f"{uid}_{mtype}_{m_seed}_model_nb_cvtest.pkl")





## check for model fit error ; try catch and then proceed with evaluation 
try:
    '''
    Call variational bayes module of STAN to obtain the model posterior
    '''
    print([l,sp_mean,sp_var, h_prop, mtype, m_seed, mtype, uid])
    NB_vb = mod.vb(data=data,iter=3000, seed = m_seed, verbose = True, \
                    adapt_engaged = True, sample_file = None, \
                    diagnostic_file = diag_file_o, eval_elbo = 50, \
                    output_samples = mtype, tol_rel_obj = tol_rel_obj_set)

    #nb_elbo = pd.read_csv(diag_file_o, comment='#', header=None)
    #nb_elbo.columns = ['iter', 'time', 'ELBO']  


    # save model output 
    #temp_fname_o = str(uid)+ '_' + str(mtype) + '_'+ str(m_seed) + '_'
    #fname_o = os.path.join(model_dir, f"{uid}_{mtype}_{m_seed}_model_nb.pkl") 
    #with open(fname_o, 'wb') as f:
    #    pickle.dump(NB_vb, f)
    #with open(fname_o, 'rb') as f:
    #    results = pickle.load(f)
        
        
    '''
    Evaluate model parameters estimate based on out of sample log-posterior predictive check [LLPD]
    Using posterior mean estimate 'mu_sample'  - generate predicted value of Y 
    Test statistics using predicted log-likelihood on the sample data 
    '''
    

    # variance estimate of  rge model parameters
    #import vb_stan as vbfun
    parma_sample  = vbfun.vb_extract_sample(NB_vb)
    parma_sample  =  dict(parma_sample)
    

    
    random.seed(m_seed)
    nsample = parma_sample['C0'].shape[0]
    mu_sample = np.zeros((nsample, n,q))
    mu_sample = mu_sample.astype(np.float64)
    parma_sample['phi'] = parma_sample['phi'].astype(np.float64)
    Yte_cv = np.zeros((nsample, n,q))
    Yte_cv = Yte_cv.astype(np.float64)
    
    ## Compute the predicted value of Y using the posterior sample.
    for s_ind in range(nsample):
        print(s_ind)
        for i in range(n):
            for j in range(q):
                # compute mean for the NB distribution 
                if holdout_mask[i,j] != 5: 
                    if mtype == 0:
                        mu_sample[s_ind, i,j] =  parma_sample['C0'][s_ind, j] + \
                            np.matmul(X[i,],parma_sample['C_geo'][s_ind,j,:]) + \
                            np.matmul(S[i,],np.matmul(parma_sample['A_s'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(Q[i,],np.matmul(parma_sample['A_m'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(B[i,],np.matmul(parma_sample['A_b'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:]));
                        
                    if mtype == 1:
                        mu_sample[s_ind, i,j] =  parma_sample['C0'][s_ind, j] + \
                            np.matmul(X[i,],parma_sample['C_geo'][s_ind,j,:]) + \
                            np.matmul(S[i,],np.matmul(parma_sample['A_s'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(Q[i,],np.matmul(parma_sample['A_m'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(B[i,],np.matmul(parma_sample['A_b'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:]));
                    if mtype == 2:
                        mu_sample[s_ind, i,j] =  parma_sample['C0'][s_ind, j] + \
                            np.matmul(S[i,],np.matmul(parma_sample['A_s'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(Q[i,],np.matmul(parma_sample['A_m'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(B[i,],np.matmul(parma_sample['A_b'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:]));
                    if mtype == 3:
                        mu_sample[s_ind, i,j] =  parma_sample['C0'][s_ind, j] + \
                            np.matmul(X[i,],parma_sample['C_geo'][s_ind,j,:]) + \
                            np.matmul(Q[i,],np.matmul(parma_sample['A_m'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(B[i,],np.matmul(parma_sample['A_b'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:]));
                    if mtype == 5:
                        mu_sample[s_ind, i,j] =  parma_sample['C0'][s_ind, j] + \
                            np.matmul(X[i,],parma_sample['C_geo'][s_ind,j,:]) + \
                            np.matmul(S[i,],np.matmul(parma_sample['A_s'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(B[i,],np.matmul(parma_sample['A_b'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:]));
                    if mtype == 4:
                        mu_sample[s_ind, i,j] =  parma_sample['C0'][s_ind, j] + \
                            np.matmul(X[i,],parma_sample['C_geo'][s_ind,j,:]) + \
                            np.matmul(S[i,],np.matmul(parma_sample['A_s'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(Q[i,],np.matmul(parma_sample['A_m'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:]));

                    if mtype == 6:
                        mu_sample[s_ind, i,j] =  parma_sample['C0'][s_ind, j] + \
                            np.matmul(X[i,],np.matmul(parma_sample['A_geo'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(S[i,],np.matmul(parma_sample['A_s'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(Q[i,],np.matmul(parma_sample['A_m'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(D[i,],np.matmul(parma_sample['A_d'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(B[i,],np.matmul(parma_sample['A_b'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])); 
                    if mtype != 1:
                        if Yi[i,j] == 1:
                            temp = Yi[i,:];temp[j] = 0;
                            mu_sample[s_ind, i,j] = mu_sample[s_ind,i,j] + np.matmul( \
                                    parma_sample['L_i'][s_ind,j,:],\
                                        np.matmul(parma_sample['L_sp'][s_ind,:,:].T,temp))/(Bs[i]-1.0);
                        
                    mu_sample[s_ind,i,j] =  data['T'][i]*np.exp(mu_sample[s_ind,i,j]* parma_sample['tau'][s_ind,j])
                    tempx1 = vbfun.neg_binomial_2_lpmf(Y[i,j], mu_sample[s_ind,i,j],\
                              1/np.sqrt(parma_sample['phi'][s_ind,j]))
                    tempx2 = vbfun.neg_binomial_2_lpmf(Y[i,j], Y[i,j],\
                              1/np.sqrt(parma_sample['phi'][s_ind,j]))
                    Yte_cv[s_ind,i,j] = tempx1


          
                    
                    

    
    
    ## get mean estimate of the posterior distribution 
    parma_mean  = dict(vbfun.vb_extract_mean(NB_vb))
    
    ## Get mean parameter estimate of the Negative Binomial distribution using the model parameters estimate         
    muest = np.zeros((n,q))
    muest1 = np.zeros((n,q))
    for i in range(n):
        for j in range(q):
            if mtype == 0:
                muest[i,j] =  parma_mean['C0'][j] + \
                    np.matmul(X[i,],parma_mean['C_geo'][j,:]) + \
                    np.matmul(S[i,],np.matmul(parma_mean['A_s'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(Q[i,],np.matmul(parma_mean['A_m'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(B[i,],np.matmul(parma_mean['A_b'],parma_mean['L_sp'][j,:]));
            if mtype == 1:
                muest[i,j] =  parma_mean['C0'][j] + \
                    np.matmul(X[i,],parma_mean['C_geo'][j,:]) + \
                    np.matmul(S[i,],np.matmul(parma_mean['A_s'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(Q[i,],np.matmul(parma_mean['A_m'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(B[i,],np.matmul(parma_mean['A_b'],parma_mean['L_sp'][j,:]));
            if mtype == 2:
                muest[i,j] =  parma_mean['C0'][j] + \
                    np.matmul(S[i,],np.matmul(parma_mean['A_s'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(Q[i,],np.matmul(parma_mean['A_m'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(B[i,],np.matmul(parma_mean['A_b'],parma_mean['L_sp'][j,:]));
            if mtype == 3:
                muest[i,j] =  parma_mean['C0'][j] + \
                    np.matmul(X[i,],parma_mean['C_geo'][j,:]) + \
                    np.matmul(Q[i,],np.matmul(parma_mean['A_m'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(B[i,],np.matmul(parma_mean['A_b'],parma_mean['L_sp'][j,:]));
            if mtype == 5:
                muest[i,j] =  parma_mean['C0'][j] + \
                    np.matmul(X[i,],parma_mean['C_geo'][j,:]) + \
                    np.matmul(S[i,],np.matmul(parma_mean['A_s'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(B[i,],np.matmul(parma_mean['A_b'],parma_mean['L_sp'][j,:]));
            if mtype == 4:
                muest[i,j] =  parma_mean['C0'][j] + \
                    np.matmul(X[i,],parma_mean['C_geo'][j,:]) + \
                    np.matmul(S[i,],np.matmul(parma_mean['A_s'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(Q[i,],np.matmul(parma_mean['A_m'],parma_mean['L_sp'][j,:]));
            if mtype == 6:
                muest[i,j] =  parma_mean['C0'][j] + \
                    np.matmul(X[i,],np.matmul(parma_mean['A_geo'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(S[i,],np.matmul(parma_mean['A_s'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(Q[i,],np.matmul(parma_mean['A_m'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(D[i,],np.matmul(parma_mean['A_d'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(B[i,],np.matmul(parma_mean['A_b'],parma_mean['L_sp'][j,:]));
            
            if mtype != 1:
                if Yi[i,j] == 1:
                    temp = Yi[i,:];temp[j] = 0;
                    muest1[i,j] = np.matmul( parma_mean['L_i'][j,:], np.matmul(parma_mean['L_sp'].T,temp))/(Bs[i]-1.0); 
                    muest[i,j] = muest[i,j] + muest1[i,j];
            muest[i,j] =  data['T'][i]*np.exp(muest[i,j]* parma_mean['tau'][j])


    ## compte log-likelihood the out of sample using the mean estimate
    Yte_fit = np.zeros((n,q))
    for i in range(n):
        for j in range(q):
            Yte_fit[i,j] = vbfun.neg_binomial_2_lpmf(Y[i,j],\
                 muest[i,j],1/np.sqrt(parma_mean['phi'][j]))
            

    ## Supporting output to compute LLPD[o] in further analysis 
    cv_test  = np.zeros((n,q))
    for i in range(n):
        print(i)
        for j in range(q):
            if holdout_mask[i,j] == 1: 
                cv_test[i,j] = np.log(np.nanmean(np.exp(Yte_cv[:,i,j])))

    Yte_fit = np.multiply(holdout_mask, Yte_fit) 
    
    # save output 
    fname_o = os.path.join(model_dir, f"{uid}_{mtype}_{m_seed}_model_nb_cvtest.pkl")  
    pickle.dump([holdout_mask, 0, 0, 0, l,m_seed,sp_mean,\
                 sp_var, h_prop, uid, mtype,\
                 Yte_fit, cv_test, Y, muest, Yte_cv, 0, 0], open(fname_o, "wb"))
except ZeroDivisionError:
    fname_o = os.path.join(model_dir, f"{uid}_{mtype}_{m_seed}_model_nb_cvtest.pkl")  
    pickle.dump([holdout_mask, 0, 0, 0, l,m_seed,sp_mean,\
                 sp_var, h_prop, uid, mtype, 0, 0,0,0,0,0,0], open(fname_o, "wb"))
    # save output flag 
    print("An exception occurred")        
    
            
        
 
    


