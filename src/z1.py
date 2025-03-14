import pandas as pd
import re

# Define file paths
z_path = "../data/data_new/Z.csv"
w1_path = "../data/data_new/W1.csv"
output_path = "../data/data_new/Z_updated.csv"

# Load files
df_z = pd.read_csv(z_path)
df_w1 = pd.read_csv(w1_path)

# 🔹 Extract only the biome code inside the first parenthesis
def extract_biome(biome_text):
    match = re.search(r"\((.*?)\)", str(biome_text))  # Find text inside parentheses
    return match.group(1) if match else None  # Return matched text or None

# Apply extraction to the `Marine pelagic biomes` column in W1
df_w1["Biome"] = df_w1["Marine pelagic biomes  (Longhurst 2007) [MRGID registered at www.marineregions.com] "].apply(extract_biome)

# 🔹 Merge Biome into Z.csv using `id` (Z) ↔ `PANGAEA sample identifier` (W1)
df_z_updated = df_z.merge(df_w1[["PANGAEA sample identifier", "Biome"]], 
                          left_on="id", right_on="PANGAEA sample identifier", how="left")

# 🔹 Drop redundant column and save the updated file
df_z_updated.drop(columns=["PANGAEA sample identifier"], inplace=True)
df_z_updated.to_csv(output_path, index=False)

print(f"✅ Updated Z.csv saved as {output_path}")

