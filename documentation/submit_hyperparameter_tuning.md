## Overview

This Python script automates the execution of hyperparameter tuning experiments. It reads a set of hyperparameter combinations from a CSV file, generates unique identifiers for each experiment run, and executes `hyperparameter_tuning_fit.py`. Redundant computations are avoided by checking for existing models. 

---
## Parameters
`n_max_run`: Number of parallel processes to run (default: 2).
`csv_path`: Path to the CSV file containing hyperparameter combinations (default: ../results/hyperparameter/hyperparams.csv).
`model_dir`: Directory where the results of each experiment are stored (default: ../results/hyperparameter/models/).
`h_prop`: Holdout proportion of the dataset for validation (default: 0.1).
`nsample_0`: Number of posterior samples to generate (default: 200).
`sid`: Fixed simulation setting identifier to associate with the runs (default: 123).
`n_repeats`: Number of times each hyperparameter combination is evaluated (default: 5).

Hyperparameters (from CSV file):

`λ`: Regularization parameter for variance. `sp_var`
`ϑ`: Regularization parameter for the mean. `sp_mean`
`k`: Latent rank. `l`


## Running 

The script will be run when using `sbatch submit_hyperparameter` and for each combination of hyperparameter in `hyperparms.csv` the `hyperparameter_tuning_fit.py` is executed. 


