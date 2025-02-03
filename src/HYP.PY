import random    
import pandas as pd
import numpy as np
from scipy.stats import norm
import pystan
import pickle
import sys
import os
import gzip
utils_dir= os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(utils_dir)
import sub_fun as sf
import vb_stan as vbfun

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "stan_model")
output_dir = os.path.join(base_dir, "results/hyperparameter/")
diag_dir = os.path.join(output_dir, "diagnostics/")
model_dir = os.path.join(output_dir, "models/")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(diag_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
# Get setting parameter for running the script
print(sys.argv)
[l,m_seed,sp_mean,sp_var, h_prop, uid, nsample_o, sid] = map(float, sys.argv[1:])
uid = int(uid); nsample_o = int(nsample_o); m_seed = int(m_seed); l = int(l)
sid = int(sid)

## Response matrix: microbial abundance data 
y_path= os.path.join(base_dir, "data/Y1.csv")
x_path = os.path.join(base_dir, "data/X.csv")
z_path = os.path.join(base_dir, "data/Z.csv")
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

'''
Prepare input data, compile stan model and define output file (to store the model output)
'''

data = {'n':Y.shape[0],'q':Y.shape[1],'p':X.shape[1],'l': l,'s':S.shape[1], \
        'b':B.shape[1], 'Y':Y, 'X':X, 'S':S, 'B':B, 'Yi':Yi, 'T':T_i, 'Bs':Bs, \
        'holdout': holdout_mask, 'sp_mean' : sp_mean, 'sp_var' : sp_var,\
        'm':Q.shape[1], 'Q': Q}

stan_mod = os.path.join(model_path, 'NB_microbe_ppc.stan')
fname = 'NB_microbe_ppc.stan'          # stan model file name
model_NB = open(stan_mod, 'r').read()     # read model file 
mod = pystan.StanModel(model_code=model_NB) # model compile 

# model output file 
sample_file_o = os.path.join(diag_dir, f"{uid}_{sid}_nb_sample.csv")
diag_file_o = os.path.join(diag_dir, f"{uid}_{sid}_nb_diag.csv")
model_output_file = os.path.join(model_dir, f"{uid}_{sid}_model_nb_cvtest.pkl")

try:
    print(f"Fitting model with l={l}, m_seed={m_seed}, sp_mean={sp_mean}, sp_var={sp_var}, h_prop={h_prop}, uid={uid}, sid={sid}")
    NB_vb = mod.vb(data=stan_data, iter=2000, seed=m_seed, verbose=True, adapt_engaged=True,
                   sample_file=os.path.join(diag_dir, f"{uid}_{sid}_nb_sample.csv"),
                   diagnostic_file=os.path.join(diag_dir, f"{uid}_{sid}_nb_diag.csv"),
                   eval_elbo=50, output_samples=nsample_o)
    
    print("VB output keys:", NB_vb.keys())
    # Extract and summarize results
    parma_mean = vbfun.vb_extract_mean(NB_vb)

    # Prepare compressed summary
    model_summary = {
        "parameter_summaries": {"mean_estimates": parma_mean},
        "diagnostics": {"elbo": NB_vb.get("elbo", None)},
        "hyperparameters": {
            "latent_rank": l,
            "sp_mean": sp_mean,
            "sp_var": sp_var,
            "holdout_proportion": h_prop
        }
    }

    # Save compressed results
    compressed_file_path = os.path.join(model_dir, f"{uid}_{sid}_summary.pkl.gz")
    with gzip.open(compressed_file_path, "wb") as f:
        pickle.dump(model_summary, f)
    print(f"Compressed and saved model summary to {compressed_file_path}")

except Exception as e:
    print(f"Error during model fitting: {e}")
