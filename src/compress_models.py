import os
import gzip
import pickle

# Paths
model_dir = "../results/hyperparameter/models/"
compressed_dir = "../results/hyperparameter/models/"  # Compressed files will overwrite originals
os.makedirs(compressed_dir, exist_ok=True)

# Get list of existing .pkl files
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]

# Function to process and compress a single model
def process_model(file_path, uid, sid):
    try:
        # Load the original model
        with open(file_path, "rb") as f:
            model_data = pickle.load(f)

        # Extract relevant parts
        model_summary = {
            "posterior_samples": {
                "C0": model_data["posterior_samples"]["C0"],  # Adjust keys as needed
                "phi": model_data["posterior_samples"]["phi"]
            },
            "parameter_summaries": model_data["parameter_summaries"],
            "diagnostics": model_data["diagnostics"],
            "hyperparameters": model_data["hyperparameters"]
        }

        # Generate the same filename format as the updated script
        compressed_file_name = f"{uid}_{sid}_summary.pkl.gz"
        compressed_file_path = os.path.join(compressed_dir, compressed_file_name)

        # Compress and save the extracted data
        with gzip.open(compressed_file_path, "wb") as f:
            pickle.dump(model_summary, f)
        
        print(f"Processed and compressed: {compressed_file_path}")

        # Optionally delete the original file to save space
        os.remove(file_path)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Process all models
for model_file in model_files:
    file_path = os.path.join(model_dir, model_file)

    # Extract uid and sid from the file name (assuming consistent naming convention like "{uid}_{sid}_model_nb.pkl")
    try:
        name_parts = model_file.split("_")
        uid = int(name_parts[0])
        sid = int(name_parts[1])
        process_model(file_path, uid, sid)
    except (IndexError, ValueError) as e:
        print(f"Skipping file {model_file}: {e}")

print("All models processed and compressed.")

