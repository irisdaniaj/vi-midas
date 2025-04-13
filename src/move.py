import os

# Path to your folder (change if needed)
folder_path = "../results/results_new_var_nc/component/models"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    parts = filename.split("_")
    
    # Only rename if the second part is "6"
    if len(parts) > 2 and parts[1] == "6":
        parts[1] = "7"
        new_filename = "_".join(parts)
        
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        print(f"Renaming: {filename} → {new_filename}")
        os.rename(old_path, new_path)

print("✅ Done renaming.")
