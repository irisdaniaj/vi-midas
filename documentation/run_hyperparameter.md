# Hyperparameter Tuning Script Documentation

## Overview
This script automates the process of running hyperparameter tuning. It reads hyperparameter combinations from `results/hyperparameter/hyperparams.csv`, generates commands to execute the tuning process, and runs 
them in parallel while ensuring that previously computed models are not reprocessed.

### Required Files and Directories
- **Hyperparameter CSV File**: `../results/hyperparameter/hyperparams.csv` containing hyperparameter combinations.
- **Model Directory**: `../results/hyperparameter/models/` to store generated models.
- **Script for Model Training**: `hyperparameter_tuning_fit.py`, which must accept the following arguments:
  - `latent rank (k)`
  - `seed_iteration`
  - `regularization mean (λ)`
  - `regularization variance (ϑ)`
  - `holdout proportion (h_prop)`
  - `simulation ID (sid_current)`
  - `number of posterior samples (nsample_0)`
  - `unique identifier (uid)`

### Key Parameters
| Parameter       | Value / Description |
|----------------|--------------------|
| `n_max_run`    | Maximum number of parallel processes (default: 12) |
| `csv_path`     | Path to the hyperparameter CSV file |
| `model_dir`    | Directory to store trained models |
| `h_prop`       | Holdout proportion (default: 0.1) |
| `nsample_0`    | Number of posterior samples (default: 200) |
| `sid`          | Simulation setting identifier (default: 123) |
| `n_repeats`    | Number of times each hyperparameter setting is repeated (default: 5) |

## Script Workflow
### 1. Setup
- Reads the `repeat` value from command-line arguments.
- Ensures the model directory exists.
- Loads hyperparameter settings from the CSV file.

### 2. Generate Commands
- Iterates through hyperparameter settings.
- Checks if the corresponding model already exists.
- If the model does not exist, a command is created to run `hyperparameter_tuning_fit.py` with appropriate parameters.

### 3. Execute Commands in Parallel
- Uses `ProcessPoolExecutor` to execute multiple tuning jobs simultaneously (up to `n_max_run`).
- Captures `stdout` and `stderr` outputs for each command.
- Logs outputs into `../results/hyperparameter/logs/` directory.

## Example Output
```
Preparing command for sid: 5, repeat: 2 (sid_current: 102)
Executing command: python3 hyperparameter_tuning_fit.py 10 2 0.5 0.3 0.1 102 200 5
Command: python3 hyperparameter_tuning_fit.py 10 2 0.5 0.3 0.1 102 200 5
Output: Model training completed successfully.
--------------------------------------------------
```

## Error Handling
- If the model file already exists, the script **skips processing**.
- If an error occurs during execution, it is logged under the `logs` directory.

## Notes
- Ensure that `hyperparameter_tuning_fit.py` is correctly implemented and executable.
- The script uses a fixed **simulation ID offset (100)** to ensure uniqueness across repeats.
- The `repeat` value must be provided as a command-line argument.

