# Hyperparameter Generation Script

## Overview

This script generates a set of hyperparameter combinations for use in hyperparameter tuning experiments. It randomly selects values for three hyperparameters: λ , ϑ , and k , which are typically used in model training processes. The generated combinations are saved into a CSV file for later use by the tuning script.

## Script Details

1. **Hyperparameter Ranges**:
    - **λ (lambda)**: A log-scaled range from 0.01 to 3000, containing 50 values. These values are chosen using `np.logspace` to cover a broad range of values that may affect model regularization.
    - **ϑ (theta)**: A linearly spaced range between 0.03125 and 0.5, containing 50 values. This range is used for model-specific hyperparameters.
    - **k (latent rank)**: A set of predefined values `[10, 16, 30, 50, 80, 100, 150, 200, 500]`, representing the possible latent ranks for matrix factorization models or similar methods.

2. **Random Selection**:
    - The script uses a random selection (`np.random.choice`) to pick one value for each hyperparameter from the predefined ranges.
    - It generates **50 unique combinations** of λ, ϑ, and k, ensuring the combinations are diverse and useful for model optimization.

3. **Saving the Hyperparameters**:
    - After generating the hyperparameter combinations, the script stores them in a CSV file, `hyperparams.csv`, which is saved in the `../results/hyperparameter/` directory.
    - Each row in the CSV corresponds to one hyperparameter combination, with columns for λ, ϑ, and k.

4. **Reproducibility**:
    - A fixed random seed (`np.random.seed(123)`) is used to ensure that the hyperparameter combinations are the same each time the script is run, enabling reproducibility of the experiment.


