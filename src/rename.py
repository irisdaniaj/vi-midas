import os
import glob
from collections import defaultdict

# Configuration
target_uid = 30
input_dir = "../results/results_new_var_c/component/models/"
pattern = "*_model_nb_cvtest.pkl"

# Track new m_seed for each mtype
mtype_counters = defaultdict(int)

# Load files
files = sorted(glob.glob(os.path.join(input_dir, pattern)))

for file_path in files:
    filename = os.path.basename(file_path)
    parts = filename.split("_")

    if len(parts) >= 4:
        old_uid = parts[0]
        mtype = parts[1]
        mseed = parts[2]

        # 🚫 Skip files with uid == 123
        if old_uid == "123":
            print(f"Skipping file (uid=123): {filename}")
            continue

        # Update counter and generate new m_seed
        mtype_counters[mtype] += 1
        new_seed = mtype_counters[mtype]

        # Rename
        new_filename = f"{target_uid}_{mtype}_{new_seed}_model_nb_cvtest.pkl"
        new_path = os.path.join(input_dir, new_filename)

        print(f"✅ Renaming: {filename} → {new_filename}")
        os.rename(file_path, new_path)
    else:
        print(f"⚠️ Skipping malformed filename: {filename}")
