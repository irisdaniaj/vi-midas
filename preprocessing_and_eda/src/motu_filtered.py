import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

# Load the data
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
count_table_file = os.path.join(data_path, 'count_table.csv')
count_df = pd.read_csv(count_table_file)
count_df = count_df.drop(columns=['Unnamed: 0'], errors='ignore')

# Shape of the original dataset
print("Original Dataset:")
print(f"  Number of samples: {count_df.shape[0]}")
print(f"  Number of unique mOTUs: {count_df.shape[1] - 1}")

# Exclude 'otu23987' 
if 'otu23987' in count_df.columns:
    count_df = count_df.drop(columns=['otu23987'])

# Shape of the original dataset
print("Original Dataset (after excluding 'otu23987'):")
print(f"  Number of samples: {count_df.shape[0]}")
print(f"  Number of unique mOTUs: {count_df.shape[1] - 1}")

# Ask the user to input the cumulative threshold
# for our project, we set it to 0.4 
try:
    cumulative_threshold = float(input("Enter the cumulative threshold (e.g., 0.4 for 40%): "))
    if not (0 <= cumulative_threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1.")
except ValueError as e:
    print(f"Invalid input: {e}")
    exit()

# Ensure 'id' is preserved
if 'id' in count_df.columns:
    id_column = count_df['id']
    numeric_df = count_df.drop(columns=['id'])
else:
    id_column = None
    numeric_df = count_df


row_sums = numeric_df.sum(axis=1)
relative_abundance_df = numeric_df.div(row_sums.replace(0, np.nan), axis=0).fillna(0) # calculate the relative abundance for each mOTU per sample


final_motus_union = set() # Initialize union of mOTUs

# Iterate over each sample
for i, row in relative_abundance_df.iterrows(): # iterate over each sample's relative abundance row
    if row.sum() == 0:  # Skip empty samples = skip the rows where the sum is zero 
        print(f"Sample {i}: No data")
        continue
    sorted_motus = row.sort_values(ascending=False) # sort mOTUs according to their contribution(s) in descending order

    cumulative_contribution = sorted_motus.cumsum() # calculating the cumulative contribution of the sorted mOTUs

    # Select mOTUs contributing up to the threshold
    motus_contributing = sorted_motus[cumulative_contribution <= cumulative_threshold].index.tolist() # select the mOTUs where the cumulatove contibution is <= current threshold

    # Handle the case where no mOTUs meet the threshold
    if not motus_contributing and not sorted_motus.empty:
        motus_contributing = [sorted_motus.index[0]]  # Include the dominant mOTU

    # Add to the union
    final_motus_union.update(motus_contributing) # update the set with the unique mOTUs contributing to the threshold

print(f"Final number of unique mOTUs retained across all samples: {len(final_motus_union)}")

# Filter the original dataset
filtered_df = count_df[list(final_motus_union)].copy() # Create a filtered DataFrame with the final set of mOTUs

# Reattach 'id' if it exists
if id_column is not None:
    filtered_df['id'] = id_column.values
    filtered_df.set_index('id', inplace=True)

# Print filtered dataset shape
print(f"Shape of filtered dataset: {filtered_df.shape}")
#print(filtered_df.head())

#output directory path and saving the filtered data 
# we are saving the motu_filtered.csv to the data folder because we need it for downstream analysis
output_dir = os.path.join(data_path, '..', 'data')
os.makedirs(output_dir, exist_ok=True)
output_path_motu = os.path.join(output_dir, "motu_filtered.csv")
filtered_df.to_csv(output_path_motu)

print(f"Data saved to {output_path_motu}")

