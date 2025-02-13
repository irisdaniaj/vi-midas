import os
import pickle
import subprocess

# Load best hyperparameters
hyperparam_file = "../notebooks/selected_hyperparam"

if not os.path.exists(hyperparam_file):
    raise FileNotFoundError(f"Error: {hyperparam_file} not found! Run hyperparameter selection first.")

with open(hyperparam_file, "rb") as f:
    best_hyperparam = pickle.load(f)

# Extract values
l = int(best_hyperparam["k"])
sp_mean = float(best_hyperparam["lambda"].iloc[0])
sp_var = float(best_hyperparam["upsilon"].iloc[0])
uid = int(best_hyperparam["uid"])
"""
print(f"sp_mean: {sp_mean} (type: {type(sp_mean)})")
print(f"sp_var: {sp_var} (type: {type(sp_var)})")
print(f"l: {l} (type: {type(l)})")
print(f"uid: {uid} (type: {type(uid)})")
"""
# Fixed parameters
h_prop = 0.1  # Holdout proportion
nsample_0 = 200  # Number of posterior samples
repeat = 1  # Random seed

# Run component contribution analysis for each component removal (mtype=0 to 5)
for mtype in range(6):
    command = f"python3 component_contribution_fit.py {l} {sp_mean} {sp_var} {h_prop} {nsample_0} {repeat} {mtype} {uid}"
    print(f"Executing: {command}")
    subprocess.run(command, shell=True)
