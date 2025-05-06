
import sys
print("Using Python at:", sys.executable)


import pandas as pd
import numpy as np
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler


# Ignore warnings
warnings.filterwarnings("ignore")

# Loading the data
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
sample_data_file = os.path.join(data_path, 'sample_data.csv')
climatology_file = os.path.join(data_path, 'climatology.csv')

# Load the sample and climatology data
sample_data_df = pd.read_csv(sample_data_file)
climatology_df = pd.read_csv(climatology_file)

# Create imputed_data directory if it doesn't exist
imputed_data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "imputed_data")
os.makedirs(imputed_data, exist_ok=True)


# Drop unnecessary index columns
sample_data_df = sample_data_df.drop(columns=['Unnamed: 0'], errors='ignore')
climatology_df = climatology_df.drop(columns=['Unnamed: 0'], errors='ignore')

# Merge the sample and climatology data on 'id'
merged_df = pd.merge(sample_data_df, climatology_df, on='id', how='inner')

# List of columns to keep
selected_columns = [
    "id", "Temperature", "Oxygen", "ChlorophyllA", "Salinity", "Fluorescence", "SST", "Chl", "PAR",
    "mld", "wind", "EKE", "Rrs555", "Rrs510", "Rrs490", "Alkalinity.total",
    "NO2", "PO4", "NO2NO3", "Si", "CO3", "Carbon.total", "Station.label", "Layer", "polar",
    "Event.date", "Latitude", "Longitude", "Ocean.region", 
]

# Create new DataFrame with only selected columns
subset_df = merged_df[selected_columns].copy()

# Function to calculate and print missingness percentage of columns
def calculate_missingness(df):
    missing_percent = df.isnull().mean() * 100
    print("Missingness (%) by column:")
    print(missing_percent[missing_percent > 0])  # Only print columns with missing values
    print("-" * 50)
    return missing_percent

# Call the function to get missingness
missing_percent = calculate_missingness(subset_df)

# Filter and sort for visualization
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

# Visualization
plt.figure(figsize=(10, 6))
missing_percent.plot(kind='bar')
plt.title("Missing Data Percentage by Column", fontsize=14)
plt.ylabel("Missing Percentage (%)")
plt.xlabel("Columns")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.show()

from sklearn.preprocessing import RobustScaler
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler

def mice_impute(df, max_iter=20, n_imputations=5):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    imputed_datasets = []

    for i in range(n_imputations):
        # Copy the original DataFrame
        df_copy = df.copy()

        # Extract only numeric data
        numeric_data = df_copy[numeric_columns].copy()

        # Track missing values
        missing_mask = numeric_data.isna()

        # Fit scaler on numeric data (ignoring NaNs)
        scaler = RobustScaler()
        scaled_data = pd.DataFrame(
            scaler.fit_transform(numeric_data),
            columns=numeric_columns,
            index=numeric_data.index
        )

        # Perform imputation
        imputer = IterativeImputer(max_iter=max_iter, random_state=42 + i)
        imputed_scaled = pd.DataFrame(
            imputer.fit_transform(scaled_data),
            columns=numeric_columns,
            index=numeric_data.index
        )

        # Inverse scaling
        imputed_unscaled = pd.DataFrame(
            scaler.inverse_transform(imputed_scaled),
            columns=numeric_columns,
            index=numeric_data.index
        )

        # Only fill in the originally missing values
        for col in numeric_columns:
            df_copy[col] = df[col].where(~missing_mask[col], imputed_unscaled[col])

        imputed_datasets.append(df_copy)

    return imputed_datasets


# Track where the original data is missing
original_missing_mask = subset_df.isna()

def simulate_and_evaluate_rmse(complete_df, missing_fraction=0.1, n_imputations=5):
    # Exclude columns you don't want to impute
    exclude_cols = ["Latitude", "Longitude"]
    numeric_columns = [col for col in complete_df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

    best_rmse = float('inf')
    best_imputed_df = None
    rmse_matrix = []
    imputed_datasets = [] 

    for i in range(n_imputations):
        missing_data = complete_df.copy()
        mask = np.random.rand(*missing_data.shape) < missing_fraction
        missing_data[mask] = np.nan

        print(f"\nImputation {i+1} - Simulated missing values: {mask.sum()}")

        imputed_data = mice_impute(missing_data, max_iter=20, n_imputations=1)[0]
        imputed_datasets.append(imputed_data) 

        rmse_values = {}

        for col in numeric_columns:
            col_mask = mask[:, complete_df.columns.get_loc(col)]
            true_values = complete_df[col][col_mask]
            imputed_values = imputed_data[col][col_mask]

            valid_mask = ~true_values.isna() & ~imputed_values.isna()
            true_values_valid = true_values[valid_mask]
            imputed_values_valid = imputed_values[valid_mask]

            if len(true_values_valid) > 0:
                rmse = np.sqrt(mean_squared_error(true_values_valid, imputed_values_valid))
                rmse_values[col] = rmse

        # Print RMSEs for this imputation
        print("RMSE by variable:")
        for col, rmse in rmse_values.items():
            print(f"  {col:20s}: {rmse:.4f}")

        rmse_matrix.append(rmse_values)

        # Update best imputed dataframe based on mean RMSE
        mean_rmse = np.mean(list(rmse_values.values()))
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_imputed_df = imputed_data

    #print(f"\n Best imputation selected (lowest mean RMSE = {best_rmse:.4f})")
    print("-" * 50)

    rmse_df = pd.DataFrame(rmse_matrix, columns=numeric_columns)
    best_imputation_indices = rmse_df.idxmin().to_dict()  # e.g., {'Oxygen': 1, 'NO2': 4, ...}

    return imputed_datasets, rmse_df, best_imputation_indices

def construct_best_variable_df(imputed_datasets, best_indices, base_df, original_missing_mask):
    best_df = base_df.copy()
    for col, best_i in best_indices.items():
        best_df[col] = base_df[col].where(~original_missing_mask[col], imputed_datasets[best_i][col])
    return best_df



import math


def plot_rmse_grid(rmse_df, best_indices=None, save_path=None, n_cols=5):
    n_vars = rmse_df.shape[1]
    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), sharey=False)
    axes = axes.flatten()

    for i, col in enumerate(rmse_df.columns):
        values = rmse_df[col]
        bar_colors = ['skyblue'] * len(values)

        # Highlight best bar
        if best_indices:
            best_idx = best_indices[col]
            bar_colors[best_idx] = 'dodgerblue'  # or 'black' if you want it bold

        axes[i].bar(range(1, len(values) + 1), values, color=bar_colors)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xticks(range(1, len(values) + 1))
        axes[i].set_xlabel("Imputation #")
        axes[i].set_ylabel("RMSE")

        y_min = values.min() * 0.9
        y_max = values.max() * 1.1
        axes[i].set_ylim(y_min, y_max)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("RMSE per Variable Across Imputations (Best Highlighted)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f" Saved RMSE plot to: {save_path}")

    plt.show()


# Set seed for reproducibility
np.random.seed(555)

# Run imputation
imputed_datasets, rmse_df, best_imputation_indices = simulate_and_evaluate_rmse(subset_df)

# Build hybrid best-per-variable DataFrame
best_hybrid_df = construct_best_variable_df(imputed_datasets, best_imputation_indices, subset_df, original_missing_mask)

columns_to_drop = ["Alkalinity.total", "CO3"] 
best_hybrid_df = best_hybrid_df.drop(columns=columns_to_drop, errors='ignore')
rmse_df = rmse_df.drop(columns=columns_to_drop, errors='ignore')
selected_columns = [col for col in selected_columns if col not in columns_to_drop]

# Plot RMSE with best bar bolded
plot_rmse_grid(rmse_df, best_indices=best_imputation_indices, save_path=os.path.join(imputed_data, "rmse_best_per_variable.png"))

# Save final hybrid dataset
best_hybrid_df[selected_columns].to_csv(os.path.join(imputed_data, "best_per_variable_imputed_data.csv"), index=False)
print("Saved best-per-variable imputed CSV.")

