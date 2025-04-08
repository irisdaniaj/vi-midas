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
#  Load Config or Parse CLI Arguments
# -------------------------
# -------------------------
#  Load Config or Parse CLI Arguments
# -------------------------
config_file = os.path.join(os.path.dirname(__file__), "config_mode.txt")
if os.path.exists(config_file):
    with open(config_file, "r") as f:
        lines = f.read().splitlines()
        data_mode = lines[0].strip() if len(lines) > 0 else "original"
        setting = int(lines[1]) if len(lines) > 1 else 1
    # Only keep the last 7 args for model parameters
    remaining_args = sys.argv[-7:]

else:
    parser = argparse.ArgumentParser(description="Run Stan Model with Original or New Data")
    parser.add_argument("--mode", choices=["original", "new"], default="original", help="Choose dataset mode: 'original' or 'new'")
    parser.add_argument("--setting", type=int, choices=[1, 2], default=1, help="Choose setting: 1 or 2")
    args, remaining_args = parser.parse_known_args()
    data_mode = args.mode
    setting = args.setting


# -------------------------
#  Set Paths Based on Data and Setting

# -------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if data_mode == "original":
    data_dir = os.path.join(base_dir, "data/data_op/")
    if setting == 1:
        stan_mod = os.path.join(base_dir, "stan_model/NB_microbe_ppc.stan")
        results_dir = os.path.join(base_dir, "results/results_old_c/sensitivity/")
    elif setting == 2:
        stan_mod = os.path.join(base_dir, "stan_model/NB_microbe_ppc_test.stan")  # ← adjust as needed
        results_dir = os.path.join(base_dir, "results/results_old_nc/sensitivity/")
elif data_mode == "new":
    data_dir = os.path.join(base_dir, "data/data_new/")
    if setting == 2:
        stan_mod = os.path.join(base_dir, "stan_model/NB_microbe_ppc_test_new.stan")  # ← adjust as needed
        results_dir = os.path.join(base_dir, "results/results_new_var_nc/sensitivity/")
    elif setting == 1:
        stan_mod = os.path.join(base_dir, "stan_model/NB_microbe_ppc_new.stan")  # ← adjust as needed
        results_dir = os.path.join(base_dir, "results/results_new_var_c/component/")


# Create necessary folders
diag_dir = os.path.join(results_dir, "diagnostics/")
model_dir = os.path.join(results_dir, "models/")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(diag_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Print confirmation
print(f"📁 Using data mode: '{data_mode}' and setting: {setting}")
print(f"📂 Data directory: {data_dir}")
print(f"📄 Stan model file: {stan_mod}")
print(f"📈 Results directory: {results_dir}")

print(sys.argv)
[l,sed,sp_mean,sp_var, h_prop, uid, nsample_o] = map(float, remaining_args)
uid = int(uid); nsample_o = int(nsample_o); m_seed = 123; l = int(l)
#m_seed = int(m_seed)

y_path= os.path.join(data_dir, "Y1.csv") #change the path of the data
x_path = os.path.join(data_dir, "X.csv")
z_path = os.path.join(data_dir, "Z.csv")
d_path = os.path.join(data_dir, "satellite.csv")
Y = pd.read_csv(y_path).to_numpy()  
Y = Y[:,range(2,Y.shape[1])]
Y = Y.astype('int')

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
X = pd.read_csv(x_path).iloc[:,1:].to_numpy()    #ADD THE VARIABLE THAT GO INTO THE GEOCHEMICAL LATENT SPACE 
X = np.subtract(X, np.mean(X, axis = 0)) # mean centering
X = X/np.std(X,axis=0)                   # scaling 


## Spatio-temporal indicators
Z = pd.read_csv(z_path)
I = Z.to_numpy()[:,range(1,Z.shape[1])]   

#satellite data AGGIUSTARE PWE IL NUOVO PATH E PENSARE COME DEVO TRASFORMARLE 
D = pd.read_csv(d_path).iloc[:,1:].to_numpy()    
D = np.subtract(D, np.mean(D, axis = 0)) # mean centering
D = D/np.std(D,axis=0)    

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
Full data analysis, model diagnostic and posterior predictive check for model validity
'''

# construct 'holdout_mask': an indicator matrix for training and testing data 
n, q = Y.shape
holdout_portion = h_prop
n_holdout = int(holdout_portion * n * q)
holdout_mask  = np.zeros(n*q)
random.seed(m_seed)
if (holdout_portion > 0.):
    tem  = np.random.choice(range(n * q), size = n_holdout, replace = False)
    holdout_mask[tem] = 1.
holdout_mask = holdout_mask.reshape((n,q))

# training and validation set for the analysis 
Y_train = np.multiply(1-holdout_mask, Y)     ## training set 
Y_vad = np.multiply(holdout_mask, Y)         ## valiation set

if data_mode == "new": 
    data = {'n':Y.shape[0],'q':Y.shape[1],'p':X.shape[1],'l': l,'s':S.shape[1], "d": D.shape[1], \
            'b':B.shape[1], 'Y':Y, 'X':X, 'S':S, 'B':B, 'Yi':Yi, 'T':T_i, 'Bs':Bs, "D": D, \
            'holdout': holdout_mask, 'sp_mean' : sp_mean, 'sp_var' : sp_var,\
            'm':Q.shape[1], 'Q': Q}
else:
    data = {'n':Y.shape[0],'q':Y.shape[1],'p':X.shape[1],'l': l,'s':S.shape[1], \
        'b':B.shape[1], 'Y':Y, 'X':X, 'S':S, 'B':B, 'Yi':Yi, 'T':T_i, 'Bs':Bs, \
        'holdout': holdout_mask, 'sp_mean' : sp_mean, 'sp_var' : sp_var,\
        'm':Q.shape[1], 'Q': Q}

model_NB = open(stan_mod, 'r').read()     # read model file 
mod = pystan.StanModel(model_code=model_NB) # model compile 

# model output file 
sample_file_o = os.path.join(diag_dir, f"{uid}_{sed}_nb_sample.csv")
diag_file_o = os.path.join(diag_dir, f"{uid}_{sed}_nb_diag.csv")
model_output_file = os.path.join(model_dir, f"{uid}_{sed}_model_nb_cvtest.pkl")

## check for model fit error ; try catch and then proceed with evaluation 
try:
    '''
    Call variational bayes module of the STAN to obtain the model posterior
    '''
    print([l,m_seed,sp_mean,sp_var, h_prop, uid, nsample_o, m_seed])
    NB_vb = mod.vb(data=data,iter=2000, seed = m_seed, verbose = True, \
                    adapt_engaged = True, sample_file = None, \
                    diagnostic_file = diag_file_o, eval_elbo = 50, \
                    output_samples = nsample_o)
    # save model output 
    fname_o = os.path.join(model_dir, f"{uid}_{sed}_model_nb.pkl")
#    with open(fname_o, 'wb') as f:
#        pickle.dump(NB_vb, f)
#    with open(fname_o, 'rb') as f:
#        results = pickle.load(f)
    '''
    Evaluate model parameters estimate based on out of sample log-posterior predictive check [LLPD]
    Using posterior mean estimate 'mu_sample'  - generate predicted value of Y 
    Test statistics using predicted log-likelihood on the sample data 
    '''
    
    
    # variance estimate of  rge model parameters

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
                if holdout_mask[i,j] == 1: 
                    # compute mean for the NB distribution 
                    if data_mode == "original" and setting == 1: 
                        mu_sample[s_ind, i,j] =  parma_sample['C0'][s_ind, j] + \
                            np.matmul(X[i,],parma_sample['C_geo'][s_ind,j,:]) + \
                            np.matmul(S[i,],np.matmul(parma_sample['A_s'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(Q[i,],np.matmul(parma_sample['A_m'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(B[i,],np.matmul(parma_sample['A_b'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:]));
                    if data_mode == "original" and setting == 2: 
                        mu_sample[s_ind, i,j] =  parma_sample['C0'][s_ind, j] + \
                            np.matmul(X[i,],np.matmul(parma_sample['A_geo'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(S[i,],np.matmul(parma_sample['A_s'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(Q[i,],np.matmul(parma_sample['A_m'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(B[i,],np.matmul(parma_sample['A_b'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:]));
                    if data_mode == "new" and setting == 2: 
                        mu_sample[s_ind, i,j] =  parma_sample['C0'][s_ind, j] + \
                            np.matmul(X[i,],np.matmul(parma_sample['A_geo'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(S[i,],np.matmul(parma_sample['A_s'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(Q[i,],np.matmul(parma_sample['A_m'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(D[i,],np.matmul(parma_sample['A_d'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(B[i,],np.matmul(parma_sample['A_b'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])); 
                    if data_mode == "new" and setting == 1: 
                        mu_sample[s_ind, i,j] =  parma_sample['C0'][s_ind, j] + \
                            np.matmul(X[i,],parma_sample['C_geo'][s_ind,j,:]) + \
                            np.matmul(S[i,],np.matmul(parma_sample['A_s'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(Q[i,],np.matmul(parma_sample['A_m'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(D[i,],np.matmul(parma_sample['A_d'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])) + \
                            np.matmul(B[i,],np.matmul(parma_sample['A_b'][s_ind,:,:],parma_sample['L_sp'][s_ind,j,:])); 
                    
                    if Yi[i,j] == 1:
                        temp = Yi[i,:];temp[j] = 0;
                        mu_sample[s_ind, i,j] = mu_sample[s_ind,i,j] + np.matmul( \
                                parma_sample['L_i'][s_ind,j,:], np.matmul(parma_sample['L_sp'][s_ind,:,:].T,temp))/(Bs[i]-1.0);
                                 
                    mu_sample[s_ind,i,j] =  data['T'][i]*np.exp(mu_sample[s_ind,i,j]* parma_sample['tau'][s_ind,j])
                    Yte_cv[s_ind,i,j] = np.exp(vbfun.neg_binomial_2_lpmf(Y[i,j], mu_sample[s_ind,i,j],\
                              1/np.sqrt(parma_sample['phi'][s_ind,j])))
                    
                        
    ## get mean estimate of the posterior distribution 
    parma_mean  = dict(vbfun.vb_extract_mean(NB_vb))

    
    ## Get mean parameter estimate of the Negative Binomial distribution using the model parameters estimate          
    muest = np.zeros((n,q))
    muest1 = np.zeros((n,q))
    for i in range(n):
        for j in range(q):
            if data_mode == "original" and setting == 1:
                muest[i,j] =  parma_mean['C0'][j] + \
                    np.matmul(X[i,],parma_mean['C_geo'][j,:]) + \
                    np.matmul(S[i,],np.matmul(parma_mean['A_s'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(Q[i,],np.matmul(parma_mean['A_m'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(B[i,],np.matmul(parma_mean['A_b'],parma_mean['L_sp'][j,:]));
            elif data_mode == "original" and setting == 2:
                muest[i,j] =  parma_mean['C0'][j] + \
                    np.matmul(X[i,],np.matmul(parma_mean['A_geo'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(S[i,],np.matmul(parma_mean['A_s'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(Q[i,],np.matmul(parma_mean['A_m'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(B[i,],np.matmul(parma_mean['A_b'],parma_mean['L_sp'][j,:]));
            if data_mode == "new" and setting == 2: 
                muest[i,j] =  parma_mean['C0'][j] + \
                    np.matmul(X[i,],np.matmul(parma_mean['A_geo'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(S[i,],np.matmul(parma_mean['A_s'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(Q[i,],np.matmul(parma_mean['A_m'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(D[i,],np.matmul(parma_mean['A_d'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(B[i,],np.matmul(parma_mean['A_b'],parma_mean['L_sp'][j,:]));
            if data_mode == "new" and setting == 1: 
                muest[i,j] =  parma_mean['C0'][j] + \
                    np.matmul(X[i,],parma_mean['C_geo'][j,:]) + \
                    np.matmul(S[i,],np.matmul(parma_mean['A_s'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(D[i,],np.matmul(parma_mean['A_d'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(Q[i,],np.matmul(parma_mean['A_m'],parma_mean['L_sp'][j,:])) + \
                    np.matmul(B[i,],np.matmul(parma_mean['A_b'],parma_mean['L_sp'][j,:]));
            
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
            
    Yte_fit = np.multiply(holdout_mask, Yte_fit) 
    
    
    ## Supporting output to compute LLPD[o] in further analysis 
    cv_test  = np.zeros((n,q))
    for i in range(n):
        print(i)
        for j in range(q):
            if holdout_mask[i,j] == 1: 
                cv_test[i,j] = np.log(np.nanmean(Yte_cv[:,i,j]))

    
    
    # save output 
    fname_o = os.path.join(model_dir, f"{uid}_{sed}_model_nb_cvtest.pkl")
    pickle.dump([holdout_mask, 0, 0, 0, l,m_seed,sp_mean,\
                 sp_var, h_prop, uid, nsample_o,\
                 Yte_fit, cv_test], open(fname_o, "wb"))
    # compute average LpmF distance
except ZeroDivisionError:
    fname_o = os.path.join(model_dir, f"{uid}_{sed}_model_nb_cvtest.pkl")
    pickle.dump([holdout_mask, 0, 0, 0, l,m_seed,sp_mean,\
                 sp_var, h_prop, uid, nsample_o, 0, 0], open(fname_o, "wb"))
    # save output flag 
    print("An exception occurred")        
    