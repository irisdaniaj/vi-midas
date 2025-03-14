import pandas as pd
import numpy as np

# Load Z.csv
z_path = "../data/data_op/Z.csv"
df_z = pd.read_csv(z_path)
I = df_z.to_numpy()[:,range(1,df_z.shape[1])]   
     
# Extract the required columns
biome_indicator = df_z.iloc[:, 0]  # Column index 1 (" ")
longhurst_province = df_z.iloc[:, 1]  # Column index 2 (EnvFeature)
quarter_indicator = df_z.iloc[:, 4]  # Column index 4 (pelagicBiomeMRGID_)

# Print extracted indicators
print("# B biome indicator")
print(biome_indicator.to_string(index=False))

print("\n# Longhurst province indicator for spatial location")
print(longhurst_province.to_string(index=False))

print("\n# Q quarter indicator for time")
print(quarter_indicator.to_string(index=False))
