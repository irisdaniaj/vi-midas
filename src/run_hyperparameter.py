import os
import pandas as pd
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

# -------------------------
#  Read Mode from Config File
# -------------------------
# Read mode from config_mode.txt
config_file = "config_mode.txt"
if os.path.exists(config_file):
    with open(config_file, "r") as f:
        lines = f.read().splitlines()
        data_mode = lines[0].strip() if len(lines) > 0 else "original"
        setting = int(lines[1]) if len(lines) > 1 else 1


# -------------------------
#  Set Paths Based on Mode
# -------------------------
if data_mode == "original" and setting == 1:
    base_results_dir = "../results/results_op/hyperparameter/"
elif data_mode == "original" and setting == 2: # ← adjust as needed
    base_results_dir =  "../results/results_new/hyperparameter/"
elif data_mode == "new" and setting == 2:
    base_results_dir= "../results/results_new_var/hyperparameter/"

#base_results_dir = "../results/results_op/hyperparameter" if data_mode == "original" else "../results/results_opd_nc/hyperparameter"
csv_path = os.path.join(base_results_dir, "hyperparams.csv")
model_dir = os.path.join(base_results_dir, "models")
log_dir = os.path.join(base_results_dir, "logs")

# Ensure directories exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


repeat = int(sys.argv[1])

# Parameters for the experiment
n_max_run = 12  # Number of parallel processes
#csv_path = "../results/hyperparameter/hyperparams.csv"
#model_dir = "../results/hyperparameter/models/"  # Directory where models are stored
h_prop = 0.1  # Holdout proportion
nsample_0 = 200  # Number of posterior samples
sid = 123  # Simulation setting identifier
n_repeats = 1  # Number of evaluations per hyperparameter combination

# Ensure the model directory exists
os.makedirs(model_dir, exist_ok=True)

# Read hyperparameter combinations from the CSV
params_df = pd.read_csv(csv_path)

# Function to check if the model file exists
def model_exists(uid, sid_current):
    model_path = os.path.join(model_dir, f"{sid_current}_{uid}_model_nb_cvtest.pkl")
    return os.path.exists(model_path)

# Dynamically generate commands from the CSV rows
commands = []
for idx, row in params_df.iterrows():
    l = int(row['k'])  # Latent rank
    sp_mean = row['λ']  # Regularization of mean parameter
    sp_var = row['ϑ']  # Regularization of variance parameter
    uid = idx  # Unique identifier for simulation settings
    seed_iteration = repeat
    # Repeat each setting n_repeats times
#    for repeat in range(1, n_repeats + 1):
    sid_current = 100 + repeat  # Create a unique identifier per repeat

    # Check if the model already exists
    if not model_exists(uid, sid_current):
        print(f"Preparing command for mode: {data_mode} setting {setting} ,sid: {uid}, repeat: {repeat} (sid_current: {sid_current})")
        cmd = f"python3 hyperparameter_tuning_fit_viet.py {data_mode} {setting} {l} {seed_iteration} {sp_mean} {sp_var} {h_prop} {sid_current} {nsample_0} {uid}"
        commands.append(cmd)
    else:
        print(f"Skipping: Model {sid_current}_{uid} already exists.")

# Function to execute a command and capture output
def run_command(command):
    try:
        print(f"Executing command: {command}")
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        return {'command': command, 'stdout': result.stdout, 'stderr': result.stderr}
    except Exception as e:
        return {'command': command, 'stdout': '', 'stderr': f"Error: {str(e)}"}

# Run commands in parallel
if commands:  # Only execute if there are commands to run
    with ProcessPoolExecutor(max_workers=n_max_run) as executor:
        results = list(executor.map(run_command, commands))
    # Print results
    for result in results:
        # Extract numeric values from the command for filename
        cmd_parts = result['command'].split()
        params_str = "_".join(cmd_parts[1:])  # Remove "python3" and keep only parameters
        log_filename = f"hyperparameter_run_{params_str}.txt"

        # Ensure safe file naming
        log_filename = log_filename.replace(".", "_")  # Replace dots with underscores
        log_filename = log_filename.replace(" ", "_")  # Remove spaces

        # Define log file path
        log_file = os.path.join(log_dir, log_filename)

        # Write log output
        with open(log_file, 'a') as f:
            f.write(f"Command: {result['command']}\n")
            f.write(f"Output: {result['stdout']}\n")
            if result['stderr']:
                f.write(f"Error: {result['stderr']}\n")
            f.close()
        print(f"Command: {result['command']}")
        print(f"Output: {result['stdout']}")
        if result['stderr']:
            print(f"Error: {result['stderr']}")
        print("-" * 50)
else:
    print("No commands to execute. All models already exist.")