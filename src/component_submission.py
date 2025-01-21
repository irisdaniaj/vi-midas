import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Submit component contribution fit jobs.")
parser.add_argument('--lambda', type=float, required=True, help="Regularization of variance parameter (λ)")
parser.add_argument('--theta', type=float, required=True, help="Regularization of mean parameter (ϑ)")
parser.add_argument('--k', type=int, required=True, help="Latent rank (k)")
args = parser.parse_args()

# Parameters for the experiment
n_max_run = 2  # Number of parallel processes
model_dir = "../results/component/models/"  # Directory where models are stored
h_prop = 0.1  # Holdout proportion
nsample_0 = 200  # Number of posterior samples
mtype_values = [0, 1, 2, 3, 4, 5]  # Model types to evaluate
uid = 123  # Fixed simulation setting identifier
n_repeats = 20  # Fixed number of evaluations per component exclusion

# Ensure the model directory exists
os.makedirs(model_dir, exist_ok=True)

# Extract the chosen hyperparameters
sp_var = args.lambda
sp_mean = args.theta
l = args.k

# Function to check if the model file exists
def model_exists(mtype, seed):
    model_path = os.path.join(model_dir, f"{uid}_{mtype}_{seed}_model_nb.pkl")
    return os.path.exists(model_path)

# Dynamically generate commands for each mtype and initialization
commands = []
for mtype in mtype_values:
    for repeat in range(1, n_repeats + 1):
        sid_current = uid * 1000 + mtype * 100 + repeat  # Create a unique identifier per repeat and mtype

        if not model_exists(mtype, repeat):
            print(f"Preparing command for mtype: {mtype}, repeat: {repeat} (sid_current: {sid_current})")
            cmd = (f"python3 component_contribution_fit.py {l} {sp_mean} {sp_var} {h_prop} "
                   f"{nsample_0} {sid_current} {mtype} {uid}")
            commands.append(cmd)
        else:
            print(f"Skipping: Model {uid}_{mtype}_{repeat} already exists.")

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
        print(f"Command: {result['command']}")
        print(f"Output: {result['stdout']}")
        if result['stderr']:
            print(f"Error: {result['stderr']}")
        print("-" * 50)
else:
    print("No commands to execute. All models already exist.")
