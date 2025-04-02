import os
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor

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
    base_results_dir = "../results/results_op/sensitivity/"
elif data_mode == "original" and setting == 2: # ← adjust as needed
    base_results_dir =  "../results/results_new/sensitivity/"
elif data_mode == "new" and setting == 2:
    base_results_dir= "../results/results_new_var/sensitivity/"

#csv_path = os.path.join(base_results_dir, "hyperparams.csv")
model_dir = os.path.join(base_results_dir, "models")
log_dir = os.path.join(base_results_dir, "logs")

# Ensure directories exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
# Paths
hyperparam_file = "../notebooks/selected_hyperparam1"
#model_dir = "../results/sensitivity/models/"
#log_dir = "../results/sensitivity/logs/"

# Ensure directories exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

if not os.path.exists(hyperparam_file):
    raise FileNotFoundError(f"Error: {hyperparam_file} not found! Run hyperparameter selection first.")

# Load hyperparameters
with open(hyperparam_file, "rb") as f:
    hyperparam_data = pickle.load(f)

# Extract values from your hyperparameter file
uid = int((hyperparam_data['uid'].iloc[0]))
l = int(hyperparam_data['k'].iloc[0])
sp_mean = float(hyperparam_data['lambda'].iloc[0])
sp_var = float(hyperparam_data['upsilon'].iloc[0])

# Fixed parameters
h_prop = 0.0  # No holdout
nsample_o = 200  # Number of posterior samples
ninit = 50 # Number of initializations
n_max_run = 12  # Limit parallel processes

# Function to check if a model already exists
def model_exists(sed):
    model_path = os.path.join(model_dir, f"{uid}_{sed}_model_nb_cvtest.pkl")
    return os.path.exists(model_path)

# Function to run a single model and log output
def run_command(sed):
    model_file = os.path.join(model_dir, f"{uid}_{sed}_model_nb_cvtest.pkl")
    log_file = os.path.join(log_dir, f"sensitivity_run_{uid}_{sed}.txt")

    # Skip model if it already exists
    if model_exists(sed):
        print(f"Skipping: Model {model_file} already exists.")
        with open(log_file, 'a') as f:
            f.write(f"Skipping: Model {model_file} already exists.\n")
        return {"command": None, "stdout": "", "stderr": ""}

    command = f"python3 model_sensitivity_fit.py {data_mode} {setting} {l} {sed} {sp_mean} {sp_var} {h_prop} {uid} {nsample_o}"
    print(f"Executing: {command}")

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Save logs for this run
        with open(log_file, 'w') as f:
            f.write(f"Command: {command}\n")
            f.write(f"Output: {result.stdout}\n")
            if result.stderr:
                f.write(f"Error: {result.stderr}\n")

        return {
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"Error: {str(e)}\n")
        return {
            "command": command,
            "stdout": "",
            "stderr": f"Error: {str(e)}"
        }

# Generate list of models that need to be computed to avoid the strange naming of the diag files 
commands_to_run = [] 
for hset in range(hyperparam_data.shape[0]):
    for init_index in range(ninit):    # Changed from 'uid' to 'init_index'
        sed = ninit*(hset + 1) + init_index
        if not model_exists(sed):
            commands_to_run.append(sed)

# Run jobs in parallel if there are models to compute
if commands_to_run:
    print(f"Running {len(commands_to_run)} models in parallel (max {n_max_run} at a time)")
    with ProcessPoolExecutor(max_workers=n_max_run) as executor:
        results = list(executor.map(run_command, commands_to_run))

    # Print results and errors
    for result in results:
        if result["command"]:  # Only print results for executed models
            print(f"Command: {result['command']}")
            print(f"Output: {result['stdout']}")
            if result['stderr']:
                print(f"Error: {result['stderr']}")
            print("-" * 50)
else:
    print("✅ All models already exist. No computation needed.")


"""
# Generate list of models that need to be computed to avoid the strange naming of the diag files 
commands_to_run = [] 
for hset in range(hyperparam_data.shape[0]):
    for init_index in range(ninit):    # Changed from 'uid' to 'init_index'
        sed = ninit*(hset + 1) + init_index
        if not model_exists(sed):
            commands_to_run.append(sed)
"""