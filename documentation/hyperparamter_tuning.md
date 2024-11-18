# Hyperparameter Tuning Script

## Overview
This script performs hyperparameter tuning for a model using **variational inference** (VI) with the Stan probabilistic programming language. The script processes input data, constructs a training and validation set, compiles the model, and saves the results for further analysis.

---

## Usage
Run the script using the command:
```bash
python hyperparameter_tuning_fit.py <l> <m_seed> <sp_mean> <sp_var> <h_prop> <uid> <nsample_o> <sid>
```

---

## Input Parameters
1. **Data Files**:
   - `./data/Y1.csv`: Response matrix (microbial abundance data).
   - `./data/X.csv`: Geochemical covariates.
   - `./data/Z.csv`: Spatio-temporal indicators.

2. **Stan Model**:
   - `./stan_model/NB_microbe_ppc.stan`: The Stan model file used for variational inference.

3. **Command-Line Arguments**:
   - `<l>`: Latent rank (integer).
   - `<m_seed>`: Seed for reproducibility (integer).
   - `<sp_mean>`: Regularization for the mean parameter (float).
   - `<sp_var>`: Regularization for the dispersion parameter (float).
   - `<h_prop>`: Proportion of data to hold out for validation (float between 0 and 1).
   - `<uid>`: Unique identifier for the simulation (integer).
   - `<nsample_o>`: Number of posterior samples to extract from the variational distribution (integer).
   - `<sid>`: Simulation setting identifier (integer).

**Example**:
```bash
python hyperparameter_tuning_fit.py 2 123 10 1 0.1 123 100 2
```

---

## Outputs
The script generates three key output files saved in the `results/hyperparameter/` directory:
1. **`<uid>_<sid>_nb_sample.csv`**:
   - Contains posterior samples generated from the variational posterior distribution.

2. **`<uid>_<sid>_nb_diag.csv`**:
   - Diagnostic information from the Stan model fitting process, including evidence lower bound (ELBO) evaluations.

3. **`<uid>_<sid>_model_nb_cvtest.pkl`**:
   - Pickled file containing:
     - Training/validation mask (`holdout_mask`).
     - Hyperparameter settings.
     - Placeholder results if errors occur during execution.

---

## Workflow
1. **Preprocessing**:
   - Reads microbial abundance data (`Y1.csv`), geochemical covariates (`X.csv`), and spatio-temporal indicators (`Z.csv`).
   - Applies geometric mean corrections and feature scaling.

2. **Model Fitting**:
   - Constructs a training-validation split based on the holdout proportion (`h_prop`).
   - Compiles and runs the Stan model (`NB_microbe_ppc.stan`) using variational inference.

3. **Output Storage**:
   - Results are saved in `results/hyperparameter/` with filenames based on `uid` and `sid`.

---

## Error Handling
- If a runtime error occurs (e.g., `ZeroDivisionError`), the script saves a placeholder result in the `.pkl` file to indicate an issue.


