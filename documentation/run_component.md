# Component Model Execution Script Documentation

## Overview
This script automates the execution of component model fitting using pre-selected hyperparameters. Parallel execution is utilized for efficiency.

### Required Files and Directories
- **Hyperparameter File**: `../notebooks/selected_hyperparam` (Pickle file containing best hyperparameter selection).
- **Model Directory**: `../results/component/models/` to store generated models.
- **Log Directory**: `../results/component/logs/` to store execution logs.
- **Script for Model Training**: `component_contribution_fit.py`.

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
| `h_prop`        | Holdout proportion (default: 0.1). |
| `nsample_0`     | Number of posterior samples (default: 200). |
| `n_max_run`     | Maximum number of parallel processes (default: 12). |

## Script Workflow
### 1. Setup
- Ensures required directories exist.
- Loads pre-selected hyperparameters from the Pickle file.
- Extracts necessary values for model execution.

### 2. Generate Commands
- Iterates through all model types (`mtype` from `0` to `5`) and seeds (`m_seed` from `0` to `19`).
- Checks if each model already exists before scheduling execution.

### 3. Execute Commands in Parallel
- Uses `ProcessPoolExecutor` to execute multiple tuning jobs simultaneously (up to `n_max_run`).
- Runs `component_contribution_fit.py` with appropriate arguments:
  ```bash
  python3 component_contribution_fit.py <l> <sp_mean> <sp_var> <h_prop> <nsample_0> <m_seed> <mtype> <uid>
  ```
- Captures `stdout` and `stderr` outputs for each command.

### 4. Logging and Results
- If a model file already exists, execution is skipped, and a message is logged.
- If execution occurs, results and errors are saved in the corresponding log file:
  ```
  ../results/component/logs/component_run_<uid>_<mtype>_<m_seed>.txt
  ```

## Example Output
```
Executing: python3 component_contribution_fit.py 10 0.5 0.3 0.1 200 3 2 123
Command: python3 component_contribution_fit.py 10 0.5 0.3 0.1 200 3 2 123
Output: Model training completed successfully.
--------------------------------------------------
```

## Error Handling
- If the model file already exists, the script **skips processing**.
- If an error occurs during execution, it is logged under the `logs` directory.
