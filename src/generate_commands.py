import os
import numpy as np
import random
random.seed(123)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_dir = os.path.join(base_dir, "results/hyperparameter/")
os.makedirs(output_dir, exist_ok=True)

# Initialize settings
seed_init = 0.     # Initial seed
test_prop = 0.1    # Proportion of test sample data
nsample_po = 200   # Number of posterior samples
nseting = 50       # Number of random hyperparameter settings

# Command for server with SLURM
command_server = 'module purge ; module load slurm gcc python3 ; OMP_NUM_THREADS=1 python3 hyperparameter_tuning_fit.py '

# Generate hyperparameter settings
setting = []
k = np.array([10, 16, 30, 50, 80, 100, 150, 200, 500])  # Latent rank
model_param = {}
model_param['mean'] = np.around(np.exp(np.random.uniform(low=np.log(0.01), 
                                                         high=np.log(3000), 
                                                         size=nseting)),
                                decimals=3)
model_param['disp'] = np.around(np.random.uniform(0.03125, 0.5, nseting), decimals=5)

# Generate commands
k_choice = np.random.choice(k, size=nseting, replace=True)
for uid in range(nseting):
    l = k_choice[uid]       # Latent rank
    j = model_param['mean'][uid]  # Lambda (mean)
    nu = model_param['disp'][uid]  # Dispersion
    for sid in range(5):     # Run each setting 5 times
        seed_init += 1
        cmd = command_server + ' '.join(list(map(str, [l, seed_init, j, nu, test_prop, uid, nsample_po, sid])))
        log_cmd = f"{cmd} > logfile/{int(seed_init)}.log 2>&1"
        setting.append(log_cmd)

# SLURM headers for array jobs
slurm_header = """#!/bin/bash
#SBATCH --array=0-249    # Adjust this range for the total number of commands
"""

# Write the commands to a SLURM script
os.makedirs("logs", exist_ok=True)  # Directory for SLURM logs
fname = os.path.join(output_dir, "mem_hyperparameter_tuning")
with open(fname, 'w') as filehandle:
    filehandle.write(slurm_header)  # Add SLURM header
    for i, place in enumerate(setting):
        filehandle.writelines(f"{place}\n")
