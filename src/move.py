import os
import shutil

# Paths (relative to the script's location in src/)
src_dir = os.path.abspath("..7results/results_new_var_c/component/models")
dst_dir = os.path.abspath("../results/results_new_var_c/component/models/")

os.makedirs(dst_dir, exist_ok=True)

# Make sure destination exists
if not os.path.exists(dst_dir):
    print(f"Destination folder does not exist: {dst_dir}")
    exit(1)

# Loop through files in source
for filename in os.listdir(src_dir):
    if not os.path.isfile(os.path.join(src_dir, filename)):
        continue  # Skip subfolders or anything that's not a file

    parts = filename.split("_")
    if len(parts) > 2 and parts[1] != "6":
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        print(f"Moving {filename} → {dst_dir}")
        shutil.move(src_path, dst_path)

print("✅ Done!")
