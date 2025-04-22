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
- Iterates through all model types (`mtype` ) and seeds (`m_seed` from `0` to `19`).
- Checks if each model already exists before scheduling execution.

| mtype  | Excluded component |
|-------|-------------|
|Original Data        |
| 0    | No component exluced  |
| 1    | Spatial: Province |
| 2    | Spatial: Depth Layer |
| 3    | Seasonal |
| 4    | Environmental |
| 5    | Species- Species Interaction |
| 6    | All variables in the latent space |
|New Data        |
| 7    |All variables in the latent space |
| 8    | No component exluced |
| 9    | Species- species interaction |
| 10   | Environmental|
| 11   | Spatial: Province |
| 12   | Spatial: Depth |
| 13   | Seasonal |
| 14   | Satellite |

# Original Data 

## Environmental Variables
- Temperature (°C)
- Salinity (PSU)
- Oxygen (µmol/Kg)
- NO2 (µmol/L)
- PO4 (µmol/L)
- NO2NO3 (µmol/L)
- Si (µmol/L)
- SST (°C)

## Spatial Variables
- Depth Layer
- Province (Biome)

## Seasonal Variable
- Seasonal (Quarter)

## Species-Species Interaction
- Interaction coefficient

# New data 
## Environmental Variables
- Temperature (°C)
- Salinity (PSU)
- Oxygen (µmol/Kg)
- NO2 (µmol/L)
- PO4 (µmol/L)
- NO2NO3 (µmol/L)
- Si (µmol/L)
- SST (°C)
- ChlorophyllA (mg/m³)
- Carbon.total (µmol/L)

## Spatial Variables
- Depth Layer
- Province (Biome)

## Seasonal Variable
- Seasonal (Quarter)

## Species-Species Interaction
- Interaction coefficient

## Satellite-Derived Variables
- Fluorescence (RFU)
- Chl (mg/m³)
- PAR (µmol photons m⁻² s⁻¹)
- mld (m)
- wind (m/s)
- EKE (m²/s²)
- Rrs490 (sr⁻¹)
- Rrs510 (sr⁻¹)
- Rrs555 (sr⁻¹)


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
