# Model Sensitivity Analysis Script Documentation

## Overview
This script automates the execution of model sensitivity analysis using predefined hyperparameters. 

## Script Execution
### Command-line Usage
This script does not require arguments and is executed directly:
```bash
python3 script_name.py
```

### Key Parameters
| Parameter        | Description |
|-----------------|-------------|
| `l`             | Latent rank from the hyperparameter file. |
| `sp_mean`       | Regularization mean (λ) from the hyperparameter file. |
| `sp_var`        | Regularization variance (ϑ) from the hyperparameter file. |
| `uid`           | Unique identifier for simulation settings. |
| `h_prop`        | Holdout proportion (default: 0.0). |
| `nsample_o`     | Number of posterior samples (default: 200). |
| `ninit`         | Number of initializations (default: 50). |
| `n_max_run`     | Maximum number of parallel processes (default: 12). |

## Script Workflow
### 1. Setup
- Ensures required directories exist.
- Loads pre-selected hyperparameters from the Pickle file.
- Extracts necessary values for model execution.

### 2. Generate Commands
- Iterates through all hyperparameter sets (`hset`) and initializations (`ninit`).
- Checks if each model already exists before scheduling execution.

### 3. Execute Commands in Parallel
- Uses `ProcessPoolExecutor` to execute multiple tuning jobs simultaneously (up to `n_max_run`).
- Runs `model_sensitivity_fit.py` with appropriate arguments:
  ```bash
  python3 model_sensitivity_fit.py <l> <sed> <sp_mean> <sp_var> <h_prop> <uid> <nsample_o>
  ```
- Captures `stdout` and `stderr` outputs for each command.

### 4. Logging and Results
- If a model file already exists, execution is skipped, and a message is logged.
- If execution occurs, results and errors are saved in the corresponding log file:
  ```
  ../results/sensitivity/logs/sensitivity_run_<uid>_<sed>.txt
  ```

## Example Output
```
Executing: python3 model_sensitivity_fit.py 10 3 0.5 0.3 0.0 123 200
Command: python3 model_sensitivity_fit.py 10 3 0.5 0.3 0.0 123 200
Output: Model training completed successfully.
--------------------------------------------------
```

## Error Handling
- If the model file already exists, the script **skips processing**.
- If an error occurs during execution, it is logged under the `logs` directory.
