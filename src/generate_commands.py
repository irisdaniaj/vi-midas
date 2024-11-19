import numpy as np
import pandas as pd

# Hyperparameter ranges
lambda_range = np.logspace(-2, 3.4771, num=50)  # Log scale for λ
theta_range = np.linspace(0.03125, 0.5, num=50)  # Linear scale for ϑ
k_values = [10, 16, 30, 50, 80, 100, 150, 200, 500]

# Randomly select 50 hyperparameter combinations
np.random.seed(42)  # For reproducibility
hyperparams = []
for i in range(50):
    l = np.random.choice(lambda_range)
    theta = np.random.choice(theta_range)
    k = np.random.choice(k_values)
    hyperparams.append({"λ": l, "ϑ": theta, "k": k})

# Save combinations to a CSV
hyperparams_df = pd.DataFrame(hyperparams)
hyperparams_df.to_csv("hyperparams.csv", index=False)
print("50 hyperparameter combinations saved to hyperparams.csv.")
