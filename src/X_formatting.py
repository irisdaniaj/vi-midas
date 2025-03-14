import pandas as pd

# Load files
x_new_path = "../data/data_new/X_new.csv"
climatology_path = "../data/data_new/climatology.csv"

df_new = pd.read_csv(x_new_path)
df_clim = pd.read_csv(climatology_path)

# 🔹 Select only the 'id' and 'SST' columns from climatology.csv
df_clim = df_clim[['id', 'SST']]

# 🔹 Merge SST into X_new using 'id' as the key
df_new_merged = df_new.merge(df_clim, on='id', how='left')

# Save modified X_new.csv
output_path = "../data/data_new/X_new.csv"
df_new_merged.to_csv(output_path, index=False)

print(f"✅ Merged X_new.csv with SST saved as {output_path}")
