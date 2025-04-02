import numpy as np
import pandas as pd
import os 
import argparse

config_file = "config_mode.txt"
if os.path.exists(config_file):
    with open(config_file, "r") as f:
        lines = f.read().splitlines()
        data_mode = lines[0].strip() if len(lines) > 0 else "original"
        setting = int(lines[1]) if len(lines) > 1 else 1

# Hyperparameter ranges
lambda_range = np.logspace(np.log10(0.01), np.log10(3000), num=50)  # Log scale for λ
theta_range = np.linspace(0.03125, 0.5, num=50)  # Linear scale for ϑ
k_values = [10, 16, 30, 50, 80, 100, 150, 200, 500]

# Randomly select 50 hyperparameter combinations
np.random.seed(123)  # For reproducibility
hyperparams = []
for i in range(50):
    l = np.random.choice(lambda_range)
    theta = np.random.choice(theta_range)
    k = np.random.choice(k_values)
    hyperparams.append({"λ": l, "ϑ": theta, "k": k})

# Define the output path
if data_mode == "original" and setting == 1:
    output_dir = "../results/results_op/hyperparameter/"
elif data_mode == "original" and setting == 2: # ← adjust as needed
    output_dir =  "../results/results_new/hyperparameter/"
elif data_mode == "new" and setting == 2:
    output_dir= "../results/results_new_var/hyperparameter/"
#output_dir = "../results/hyperparameter"

# Check if the results folder exists, create it if not
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory {output_dir} created.")

# Save combinations to a CSV
hyperparams_df = pd.DataFrame(hyperparams)
hyperparams_df.to_csv(os.path.join(output_dir, "hyperparams.csv"), index=False)

print("50 hyperparameter combinations saved to hyperparams.csv.")
