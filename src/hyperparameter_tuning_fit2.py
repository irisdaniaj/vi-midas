import os
import sys
import random
import pandas as pd
import numpy as np
import pystan
import pickle
import gzip

# Import custom modules
utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(utils_dir)
import sub_fun as sf
import vb_stan as vbfun

# Set paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "stan_model")
output_dir = os.path.join(base_dir, "results/hyperparameter/")
diag_dir = os.path.join(output_dir, "diagnostics/")
model_dir = os.path.join(output_dir, "models/")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(diag_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Parse command-line arguments
[l, m_seed, sp_mean, sp_var, h_prop, uid, nsample_o, sid] = map(float, sys.argv[1:])
uid = int(uid)
nsample_o = int(nsample_o)
m_seed = int(m_seed)
l = int(l)
sid = int(sid)

# Load data
y_path = os.path.join(base_dir, "data/Y1.csv")
x_path = os.path.join(base_dir, "data/X.csv")
z_path = os.path.join(base_dir, "data/Z.csv")
Y = pd.read_csv(y_path).to_numpy()[:, 2:].astype('int')
X = pd.read_csv(x_path).iloc[:, 1:].to_numpy()
Z = pd.read_csv(z_path).to_numpy()[:, 1:]

# Preprocess data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
errx = 1e-5
delta = np.array([sf.get_geomean(row, errx) for row in Y])
delta = delta.min()  # Ensure a scalar value for delta
T_i = np.exp(np.mean(np.log(Y + delta), axis=1))
Y = Y + delta
Y = Y.astype('int')

# Prepare indicators
n, q = Y.shape
fac = np.unique(Z[:, 0])
B = np.zeros((n, len(fac)))
for i, f in enumerate(fac):
    B[Z[:, 0] == f, i] = 1

fac = np.unique(Z[:, 1])
S = np.zeros((n, len(fac)))
for i, f in enumerate(fac):
    S[Z[:, 1] == f, i] = 1

fac = np.unique(Z[:, 4])
Q = np.zeros((n, len(fac)))
for i, f in enumerate(fac):
    Q[Z[:, 4] == f, i] = 1

# Create holdout mask
holdout_portion = h_prop
n_holdout = int(holdout_portion * n * q)
random.seed(m_seed)
holdout_mask = np.zeros(n * q)
holdout_mask[np.random.choice(range(n * q), size=n_holdout, replace=False)] = 1
holdout_mask = holdout_mask.reshape((n, q))

# Compute Bs (count of non-zero entries in rows of Y)
Bs = np.sum(Y != 0, axis=1)

# Define Stan data
stan_data = {
    'n': n, 'q': q, 'p': X.shape[1], 'l': l, 's': S.shape[1],
    'b': B.shape[1], 'Y': Y, 'X': X, 'S': S, 'B': B, 'Yi': (Y != 0) + 0, 'T': T_i,
    'holdout': holdout_mask, 'sp_mean': sp_mean, 'sp_var': sp_var,
    'm': Q.shape[1], 'Q': Q, 'Bs': Bs
}

# Compile and fit Stan model
stan_mod = os.path.join(model_path, 'NB_microbe_ppc.stan')
model_code = open(stan_mod, 'r').read()
mod = pystan.StanModel(model_code=model_code)

try:
    print(f"Fitting model with l={l}, m_seed={m_seed}, sp_mean={sp_mean}, sp_var={sp_var}, h_prop={h_prop}, uid={uid}, sid={sid}")
    NB_vb = mod.vb(
        data=stan_data,
        iter=2000,
        seed=m_seed,
        verbose=True,
        adapt_engaged=True,
        sample_file=os.path.join(diag_dir, f"{uid}_{sid}_nb_sample.csv"),
        diagnostic_file=os.path.join(diag_dir, f"{uid}_{sid}_nb_diag.csv"),
        eval_elbo=50,
        output_samples=nsample_o
    )

    # Extract the mean estimates of the parameters
    parma_mean = vbfun.vb_extract_mean(NB_vb)

    # Path to the diagnostic file
    diag_file_path = os.path.join(diag_dir, f"{uid}_{sid}_nb_diag.csv")
    final_elbo = None  # Initialize ELBO value

    # Parse the diagnostic file to retrieve the final ELBO
    if os.path.exists(diag_file_path):
        with open(diag_file_path, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    iter_num, time_in_seconds, elbo = line.strip().split(",")
                    final_elbo = float(elbo)  # Keep the last ELBO value
    else:
        print(f"Diagnostic file not found: {diag_file_path}")

    # Prepare compressed summary
    model_summary = {
        "parameter_summaries": {"mean_estimates": parma_mean},
        "diagnostics": {"elbo": final_elbo},
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

"""
try:
    print(f"Fitting model with l={l}, m_seed={m_seed}, sp_mean={sp_mean}, sp_var={sp_var}, h_prop={h_prop}, uid={uid}, sid={sid}")
    NB_vb = mod.vb(data=stan_data, iter=2000, seed=m_seed, verbose=True, adapt_engaged=True,
                   sample_file=os.path.join(diag_dir, f"{uid}_{sid}_nb_sample.csv"),
                   diagnostic_file=os.path.join(diag_dir, f"{uid}_{sid}_nb_diag.csv"),
                   eval_elbo=50, output_samples=nsample_o)
    
    #print("VB output keys:", NB_vb.keys())
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
"""