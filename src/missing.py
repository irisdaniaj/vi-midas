import pandas as pd

# Define file path
z_updated_path = "../data/data_new/Z_updated.csv"

# Load the file
df_z = pd.read_csv(z_updated_path)

# 🔹 Count missing values in the "Biome" column
missing_count = df_z["Biome"].isna().sum()

print(f"✅ Missing values in 'Biome' column: {missing_count}")
