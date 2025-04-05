import os
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor

# -------------------------
#  Read Mode from Config File
# -------------------------
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
    base_results_dir = "../results/results_op/component/"
elif data_mode == "original" and setting == 2: # ← adjust as needed
    base_results_dir =  "../results/results_new/component/"
elif data_mode == "new" and setting == 2:
    base_results_dir= "../results/results_new_var/component/"
model_dir = os.path.join(base_results_dir, "models")
log_dir = os.path.join(base_results_dir, "logs")

# Ensure directories exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
# Paths
hyperparam_file = f"../notebooks/selected_hyperparam_s{setting}"
#model_dir = "../results/component/models/"
#log_dir = "../results/component/logs/"

# Ensure directories exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

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

# Fixed parameters
h_prop = 0.1  # Holdout proportion
nsample_0 = 200  # Number of posterior samples
n_max_run = 12  # Limit parallel processes

# Function to check if a model already exists
def model_exists(mtype, m_seed):
    model_path = os.path.join(model_dir, f"{uid}_{mtype}_{m_seed}_model_nb_cvtest.pkl")
    return os.path.exists(model_path)

# Function to run a single model and log output
def run_command(params):
    mtype, m_seed = params  # Unpack model type and seed
    model_file = os.path.join(model_dir, f"{uid}_{mtype}_{m_seed}_model_nb_cvtest.pkl")
    log_file = os.path.join(log_dir, f"component_run_{uid}_{mtype}_{m_seed}.txt")

    # Skip model if it already exists
    if model_exists(mtype, m_seed):
        print(f"Skipping: Model {model_file} already exists.")
        with open(log_file, 'a') as f:
            f.write(f"Skipping: Model {model_file} already exists.\n")
        return {"command": None, "stdout": "", "stderr": ""}

    command = f"python3 component_contribution_fit.py {data_mode} {setting} {l} {sp_mean} {sp_var} {h_prop} {nsample_0} {m_seed} {mtype}"
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

# **Step 1: Generate list of models that need to be computed**
# 🔹 Ensure only mtype = 6 runs when mode is "new"
if data_mode == "new" and setting == 2:
    commands_to_run = [
        (7, m_seed) for m_seed in range(20) if not model_exists(7, m_seed)
    ]
elif data_mode == "new" and setting == 1:
    commands_to_run = [
        (mtype, m_seed) for mtype in range(8, 15) for m_seed in range(20)
        if not model_exists(mtype, m_seed)
    ]
elif data_mode == "original" and setting == 2:
    commands_to_run = [
        (6, m_seed) for m_seed in range(6) for m_seed in range(20)
        if not model_exists(6, m_seed)
    ]
elif data_mode == "original" and setting == 1: 
    commands_to_run = [
        (mtype, m_seed) for mtype in range(6) for m_seed in range(20)
        if not model_exists(mtype, m_seed)
    ]


# **Step 2: Run jobs in parallel if there are models to compute**
if commands_to_run:
    with ProcessPoolExecutor(max_workers=n_max_run) as executor:
        results = list(executor.map(run_command, commands_to_run))

    # **Step 3: Print results and errors**
    for result in results:
        if result["command"]:  # Only print results for executed models
            print(f"Command: {result['command']}")
            print(f"Output: {result['stdout']}")
            if result['stderr']:
                print(f"Error: {result['stderr']}")
            print("-" * 50)
else:
    print("✅ All models already exist. No computation needed.")
