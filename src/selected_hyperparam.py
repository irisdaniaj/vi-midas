import os
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor


# Paths
hyperparam_file = "eesha/selected_hyperparam"
#model_dir = "../results/component/models/"
#log_dir = "../results/component/logs/"


if not os.path.exists(hyperparam_file):
    raise FileNotFoundError(f"Error: {hyperparam_file} not found! Run hyperparameter selection first.")

# Load best hyperparameters
with open(hyperparam_file, "rb") as f:
    best_hyperparam = pickle.load(f)

# Extract values
l = int(best_hyperparam["k"])
sp_mean = float(best_hyperparam["lambda"].iloc[0])
sp_var = float(best_hyperparam["upsilon"].iloc[0])
uid = int(best_hyperparam["uid"])

print("l", l)
print("sp_mean", sp_mean)
print("sp_var", sp_var)
print("uid", uid)