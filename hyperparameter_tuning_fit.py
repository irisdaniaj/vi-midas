#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:42:08 2020

@author: amishra
"""

import os
import random    
import pandas as pd
import numpy as np
from scipy.stats import norm
import pystan
import pickle
import sys

# Ensure the results directory exists
output_dir = "results/hyperparameter/"
os.makedirs(output_dir, exist_ok=True)

# Get setting parameter for running the script
print(sys.argv)
[l, m_seed, sp_mean, sp_var, h_prop, uid, nsample_o, sid] = map(float, sys.argv[1:])
uid = int(uid); nsample_o = int(nsample_o); m_seed = int(m_seed); l = int(l)
sid = int(sid)

'''
# lanent_rank [l]; model seed [m_seed] 
# regularization of the mean parameter [sp_mean]
# regularization of the dispersion parameter [sp_var]
# holdout proporion of the test sample [h_prop]
# number of posterior samples from the variational posterior distribution [nsample_o]
# identifier for the simulation seting: uid
# identifier for the selected seting: sid
'''
## local test setting  
#l = 2; m_seed = 123;  sp_mean = 10;  sp_var = 1; 
#h_prop = 0.1;nsample_o = 100; uid = 123; sid = 2



# Import data for model fitting
Y = pd.read_csv('./data/Y1.csv').to_numpy()  
Y = Y[:, range(2, Y.shape[1])]
Y = Y.astype('int')

# Computation of the geometric mean
import sub_fun as sf
errx = 1e-5
delta = np.empty(Y.shape[0])  
for i in range(Y.shape[0]):
    delta[i] = sf.get_geomean(Y[i], errx)
    
T = np.exp(np.mean(np.log(Y + delta.min()), axis=1))
Bs = np.sum(Y != 0, axis=1)
Yi = (Y != 0) + 0

# Correction for the geometric mean 
T_i = np.exp(np.mean(np.log(Y.T + delta), axis=0))
Y = (Y.T + delta).T
Y = Y.astype('int')

# Geochemical covariates 
X = pd.read_csv('./data/X.csv').iloc[:, 1:].to_numpy()    
X = np.subtract(X, np.mean(X, axis=0)) # mean centering
X = X / np.std(X, axis=0)              # scaling 

# Spatio-temporal indicators
Z = pd.read_csv('./data/Z.csv')
I = Z.to_numpy()[:, range(1, Z.shape[1])]

# B biome indicator 
Ifac = I[:, 0]
fac = np.unique(Ifac)
B = np.zeros((X.shape[0], fac.shape[0]))
for i in range(fac.shape[0]):
    B[np.where(Ifac == fac[i]), i] = 1
    
# Longhurst province indicator for spatial location
Ifac = I[:, 1]
fac = np.unique(Ifac)
S = np.zeros((X.shape[0], fac.shape[0]))
for i in range(fac.shape[0]):
    S[np.where(Ifac == fac[i]), i] = 1

# Q quarter indicator for time
Ifac = I[:, 4]
fac = np.unique(Ifac)
Q = np.zeros((X.shape[0], fac.shape[0]))
for i in range(fac.shape[0]):
    Q[np.where(Ifac == fac[i]), i] = 1

# Construct 'holdout_mask': an indicator matrix for training and testing data 
n, q = Y.shape
holdout_portion = h_prop
n_holdout = int(holdout_portion * n * q)
holdout_mask = np.zeros(n * q)
random.seed(m_seed)
if (holdout_portion > 0.):
    tem = np.random.choice(range(n * q), size=n_holdout, replace=False)
    holdout_mask[tem] = 1.
holdout_mask = holdout_mask.reshape((n, q))

# Training and validation set for the analysis 
Y_train = np.multiply(1 - holdout_mask, Y)     # training set 
Y_vad = np.multiply(holdout_mask, Y)           # validation set

# Prepare input data, compile stan model and define output file
data = {'n': Y.shape[0], 'q': Y.shape[1], 'p': X.shape[1], 'l': l, 's': S.shape[1], 
        'b': B.shape[1], 'Y': Y, 'X': X, 'S': S, 'B': B, 'Yi': Yi, 'T': T_i, 'Bs': Bs,
        'holdout': holdout_mask, 'sp_mean': sp_mean, 'sp_var': sp_var,
        'm': Q.shape[1], 'Q': Q}

fname = './stan_model/NB_microbe_ppc.stan'  # stan model file name
model_NB = open(fname, 'r').read()  # read model file 
mod = pystan.StanModel(model_code=model_NB)  # model compile 

# Define output filenames in the results folder
sample_file_o = os.path.join(output_dir, f"{uid}_{sid}_nb_sample.csv")
diag_file_o = os.path.join(output_dir, f"{uid}_{sid}_nb_diag.csv")
model_output_file = os.path.join(output_dir, f"{uid}_{sid}_model_nb_cvtest.pkl")

# Check for model fit error, try catch and proceed with evaluation 
try:
    print([l, m_seed, sp_mean, sp_var, h_prop, uid, nsample_o, sid])
    NB_vb = mod.vb(data=data, iter=2000, seed=m_seed, verbose=True,
                   adapt_engaged=True, sample_file=sample_file_o,
                   diagnostic_file=diag_file_o, eval_elbo=50,
                   output_samples=nsample_o)

    # Save model output 
    with open(model_output_file, 'wb') as f:
        pickle.dump([holdout_mask, 0, 0, 0, l, m_seed, sp_mean,
                     sp_var, h_prop, uid, nsample_o, None, None], f)

except ZeroDivisionError:
    # Save a placeholder output if an error occurs
    with open(model_output_file, 'wb') as f:
        pickle.dump([holdout_mask, 0, 0, 0, l, m_seed, sp_mean,
                     sp_var, h_prop, uid, nsample_o, 0, 0], f)
    print("An exception occurred")
