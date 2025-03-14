import tarfile
import os

# Define file paths
compressed_file = "../results/results_op/component/models/models_compressed.tar.gz"  # Update the path if needed
extract_to = "../results/results_op/component/models/extracted"  # Folder where files will be extracted

# Ensure the output folder exists
os.makedirs(extract_to, exist_ok=True)

# Extract the archive
with tarfile.open(compressed_file, "r:gz") as tar:
    tar.extractall(path=extract_to)
    print(f"✅ Extracted files to: {extract_to}")
