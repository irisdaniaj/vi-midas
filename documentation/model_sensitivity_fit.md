# Variational Bayes Model Fitting Script Documentation

## Overview
This Python script performs variational inference for a Negative Binomial model using microbial abundance data. 
The model incorporates geochemical covariates, spatial and temporal indicators, and biome information to estimate microbial distributions.

## Script Execution
### Command-line Usage
```bash
python3 script_name.py <l> <m_seed> <sp_mean> <sp_var> <h_prop> <uid> <nsample_o>
```
Where:
- `<l>`: Latent rank.
- `<m_seed>`: Random seed for reproducibility.
- `<sp_mean>`: Regularization for the mean parameter.
- `<sp_var>`: Regularization for the dispersion parameter.
- `<h_prop>`: Holdout proportion for test samples.
- `<uid>`: Unique identifier for the simulation setting.
- `<nsample_o>`: Number of posterior samples from the variational posterior distribution.

## Script Workflow
### 1. Setup
- Ensures required directories exist.
- Loads data from CSV files (`Y1.csv`, `X.csv`, `Z.csv`).
- Applies transformations to response variables and covariates.
- Constructs spatial (`S`), biome (`B`), and temporal (`Q`) indicators.

### 2. Data Preprocessing
- Computes the geometric mean of microbial abundance data.
- Normalizes geochemical covariates by mean centering and scaling.
- Constructs indicator matrices for biome, location, and temporal components.

### 3. Holdout Sample Selection
- Generates a holdout mask for test samples.
- Splits data into training (`Y_train`) and validation (`Y_vad`) sets.

### 4. Model Compilation and Fitting
- Reads the `NB_microbe_ppc.stan` model file.
- Compiles the Stan model using PyStan.
- Runs variational inference with:
  - `iter=2000`
  - `adapt_engaged=True`
  - `eval_elbo=50`
  - `output_samples=<nsample_o>`

### 5. Model Output and Diagnostics
- Saves model samples to `diagnostics/` and `models/` directories.
- Extracts parameter estimates and performs posterior predictive checks.
- Computes predicted microbial abundance values.

### 6. Error Handling
- If a `ZeroDivisionError` occurs, it saves default output files with placeholders.

## Output Files
| File Name | Description |
|-----------|-------------|
| `results/sensitivity/models/<uid>_<m_seed>_model_nb_cvtest.pkl` | Trained model output. |
| `results/sensitivity/models/<uid>_<m_seed>_sample_model_nb_cvtest.pkl` | Posterior samples. |
| `results/sensitivity/diagnostics/<uid>_<m_seed>_nb_diag.csv` | Diagnostic output. |

