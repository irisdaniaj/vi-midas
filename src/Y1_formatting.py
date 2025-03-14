import pandas as pd

# Define file paths
y1_path = "../data/data_new/Y1_new.csv"
output_path = "../data/data_new/Y1_final.csv"

# Load the file
df_y1 = pd.read_csv(y1_path, dtype=str)  # Load everything as strings

# 🔹 Rename the first column as `""`
df_y1.rename(columns={df_y1.columns[0]: '""'}, inplace=True)

# 🔹 Ensure all other column names are in uppercase and properly quoted
df_y1.columns = [f'"{col.upper()}"' if col != '""' else col for col in df_y1.columns]

# 🔹 Add quotes to the first entry of every row
df_y1.iloc[:, 0] = df_y1.iloc[:, 0].apply(lambda x: f'"{x}"')

# Save the formatted file
df_y1.to_csv(output_path, index=False)

print(f"✅ Final formatted Y1.csv saved as {output_path}")
