import os
import gzip
import pickle
import pandas as pd

# Directories
dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
base_dir = os.path.join(dir, "results/hyperparameter/")
diag_dir = os.path.join(base_dir, "diagnostics")
model_dir = os.path.join(base_dir, "models")

# Collect diagnostic and model files
diag_files = [f for f in os.listdir(diag_dir) if f.endswith("_nb_diag.csv")]
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl.gz")]

# Map diagnostic files to models using UID and SID
for model_file in model_files:
    # Extract UID and SID from the model filename
    uid, sid, _ = model_file.split("_")
    diag_file = f"{uid}_{sid}_nb_diag.csv"  # Construct expected diagnostic file name
    
    diag_file_path = os.path.join(diag_dir, diag_file)
    model_file_path = os.path.join(model_dir, model_file)

    if not os.path.exists(diag_file_path):
        print(f"Diagnostic file not found for model {model_file}. Skipping...")
        continue

    # Parse the diagnostic file to get the final ELBO
    final_elbo = None
    with open(diag_file_path, "r") as f:
        for line in f:
            if not line.startswith("#"):  # Skip header lines
                _, _, elbo = line.strip().split(",")
                final_elbo = float(elbo)  # Keep the last ELBO value

    # Load the model summary
    with gzip.open(model_file_path, "rb") as f:
        model_summary = pickle.load(f)

    # Update the diagnostics with the final ELBO
    model_summary["diagnostics"]["elbo"] = final_elbo

    # Save the updated model summary
    with gzip.open(model_file_path, "wb") as f:
        pickle.dump(model_summary, f)

    print(f"Updated model summary for {model_file} with ELBO = {final_elbo}")
