#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import numpy as np
import pandas as pd
import pystan

# --- Define Paths ---
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_dir = os.path.join(base_dir, "results/results_op/sensitivity/")
diag_dir = os.path.join(results_dir, "diagnostics/")
model_dir = os.path.join(results_dir, "models/")
stan_model_path = os.path.join(base_dir, "stan_model/NB_microbe_ppc.stan")  # Ensure this exists

# --- Define Model Parameters ---
uid, m_seed = 30, 68  # ⚠️ Ensure these match the model ID we are recalculating
l = 100  # Rank from best_setting
sp_mean = 0.060596  # Lambda from best_setting
sp_var = 0.040816  # Upsilon from best_setting
h_prop = 0.0  # Holdout proportion (assuming it's 0.0)
nsample_o = 100
# --- Load Data ---
data_dir = os.path.join(base_dir, "data/data_op/")
Y = pd.read_csv(os.path.join(data_dir, "Y1.csv")).to_numpy()
X = pd.read_csv(os.path.join(data_dir, "X.csv")).iloc[:, 1:].to_numpy()
Z = pd.read_csv(os.path.join(data_dir, "Z.csv")).to_numpy()[:, 1:]

Y = pd.read_csv('Y1.csv').to_numpy()  
Y = Y[:,range(2,Y.shape[1])]
Y = Y.astype('int')

## Computation of the geometric mean:  
import sub_fun as sf
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
X = pd.read_csv('X.csv').iloc[:,1:].to_numpy()    
X = np.subtract(X, np.mean(X, axis = 0)) # mean centering
X = X/np.std(X,axis=0)                   # scaling 


## Spatio-temporal indicators
Z = pd.read_csv('Z.csv')
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
    
    
    
# --- Prepare Holdout Mask ---
n, q = Y.shape
n_holdout = int(h_prop * n * q)
holdout_mask = np.zeros(n * q)
np.random.seed(m_seed)
if n_holdout > 0:
    holdout_mask[np.random.choice(range(n * q), size=n_holdout, replace=False)] = 1
holdout_mask = holdout_mask.reshape((n, q))

# --- Prepare Stan Data ---
# --- Prepare Stan Data ---
stan_data = {'n':Y.shape[0],'q':Y.shape[1],'p':X.shape[1],'l': l,'s':S.shape[1], \
        'b':B.shape[1], 'Y':Y, 'X':X, 'S':S, 'B':B, 'Yi':Yi, 'T':T_i, 'Bs':Bs, \
        'holdout': holdout_mask, 'sp_mean' : sp_mean, 'sp_var' : sp_var,\
        'm':Q.shape[1], 'Q': Q}

# --- Compile & Run Stan Model ---
print("✅ Compiling Stan model...")
if not os.path.exists(stan_model_path):
    raise FileNotFoundError(f"❌ Stan model not found at: {stan_model_path}")

with open(stan_model_path, "r") as f:
    model_code = f.read()

stan_model = pystan.StanModel(model_code=model_code)
print("✅ Running Variational Bayes (VB)...")
stan_results = stan_model.vb(data=stan_data, iter=2000, seed=m_seed, output_samples=nsample_o)

# --- Save Full Model Output (`model_nb.pkl`) ---
model_nb_file = os.path.join(model_dir, f"{uid}_{m_seed}_model_nb.pkl")
with open(model_nb_file, "wb") as f:
    pickle.dump(stan_results, f)
print(f"✅ Saved: {model_nb_file}")

# --- Extract Variational Bayes Diagnostics ---
elbo_values = np.array([float(line.split(",")[-1]) for line in stan_results["args"]["sample_file"].split("\n") if line and not line.startswith("#")])
df_diag = pd.DataFrame({"iter": np.arange(len(elbo_values)) * 50, "ELBO": elbo_values})

# --- Save Diagnostics (`nb_diag.csv`) ---
nb_diag_file = os.path.join(diag_dir, f"{uid}_{m_seed}_nb_diag.csv")
df_diag.to_csv(nb_diag_file, index=False)
print(f"✅ Saved: {nb_diag_file}")

# --- Compute LLPD and Predictions ---
parma_sample = dict(stan_results["sampler_params"])
mu_sample = np.zeros((nsample_o, n, q))

# Compute predicted values
for s_ind in range(nsample_o):
    for i in range(n):
        for j in range(q):
            mu_sample[s_ind, i, j] = (parma_sample["C0"][s_ind, j] +
                                      np.dot(X[i,], parma_sample["C_geo"][s_ind, j, :]) +
                                      np.dot(S[i,], np.dot(parma_sample["A_s"][s_ind, :, :], parma_sample["L_sp"][s_ind, j, :])) +
                                      np.dot(Q[i,], np.dot(parma_sample["A_m"][s_ind, :, :], parma_sample["L_sp"][s_ind, j, :])) +
                                      np.dot(B[i,], np.dot(parma_sample["A_b"][s_ind, :, :], parma_sample["L_sp"][s_ind, j, :])))
mu_sample = np.exp(mu_sample)

Yte_sample = np.random.poisson(mu_sample)
Yte_cv = np.exp(np.log(mu_sample))  # Log probability mass function

# Compute Log-Likelihood Per Data Point (LLPD)
cv_test = np.zeros((n, q))
for i in range(n):
    for j in range(q):
        if holdout_mask[i, j] == 0:
            cv_test[i, j] = np.log(np.nanmean(Yte_cv[:, i, j]))

            

# --- Save LLPD Results (`model_nb_cvtest.pkl`) ---
model_nb_cvtest_file = os.path.join(model_dir, f"{uid}_{m_seed}_model_nb_cvtest.pkl")
with open(model_nb_cvtest_file, "wb") as f:
    pickle.dump([holdout_mask, 0, 0, l, m_seed, sp_mean, sp_var, h_prop, uid, nsample_o, Yte_sample, cv_test], f)
print(f"✅ Saved: {model_nb_cvtest_file}")

print("\n🎉✅ All files successfully recalculated and saved!")
