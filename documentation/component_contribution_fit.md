# Overview 

The script evaluates the contribution of individual components in a generative model for microbial abundance data. It does this by systematically excluding components from the model and estimating the variational posterior, comparing the results using log-likelihood predictive density (LLPD).


### **Purpose**
The script:
1. Fits a model with or without specific components (e.g., geochemical, spatio-temporal, species interactions).
2. Uses variational inference to approximate the posterior distribution of the model parameters.
3. Evaluates the model's performance using out-of-sample log-likelihood and saves results for further analysis.

---

### **Inputs**
The script accepts the following command-line arguments:

| Argument           | Type   | Description                                                                 |
|--------------------|--------|-----------------------------------------------------------------------------|
| `l`                | int    | Latent rank (dimensionality of the latent variable space).                  |
| `sp_mean`          | float  | Regularization parameter for the mean.                                     |
| `sp_var`           | float  | Regularization parameter for the variance.                                 |
| `h_prop`           | float  | Proportion of data used as a holdout sample for evaluation.                |
| `nsample_o`        | int    | Number of posterior samples to generate.                                   |
| `m_seed`           | int    | Seed for random initialization.                                            |
| `mtype`            | int    | Model type, specifying the component excluded (0 = full model, others vary).|
| `uid`              | int    | Unique identifier for the experiment (simulation setting).                 |

Example command:
```bash
python component_contribution_fit.py 10 398.199 0.04904 0.1 200 123 0 1
```

---

### **Key Features**

1. **Component-Excluded Models:**
   - The script uses different Stan models (`NB_microbe_ppc.stan`, `NB_microbe_ppc_nointer.stan`, etc.) based on `mtype`:
     | `mtype` | Model Description                     |
     |---------|---------------------------------------|
     | `0`     | Full model (baseline).               |
     | `1`     | Excludes species-species interactions.|
     | `2`     | Excludes geochemical data.           |
     | `3`     | Excludes spatial province data.      |
     | `4`     | Excludes biome data.                 |
     | `5`     | Excludes temporal data.              |

2. **Data Preprocessing:**
   - Reads microbial abundance (`Y`), geochemical covariates (`X`), and spatio-temporal indicators (`Z`) from CSV files.
   - Normalizes data (e.g., mean-centering, scaling).

3. **Holdout Evaluation:**
   - Creates a holdout mask to split data into training and validation sets.
   - Uses this to calculate out-of-sample LLPD for performance evaluation.

4. **Variational Inference:**
   - Fits the model using `pystan` with variational Bayes (`mod.vb()`).
   - Computes posterior distributions and diagnostics.

5. **Outputs:**
   - Saves posterior samples and evaluation metrics to `.pkl` files:
     - Posterior: `"{uid}_{mtype}_{m_seed}_model_nb.pkl"`.
     - Cross-validation results: `"{uid}_{mtype}_{m_seed}_model_nb_cvtest.pkl"`.

---

### **Workflow**

1. **Command-Line Argument Parsing:**
   - Extracts arguments for hyperparameters, model type, and simulation settings.

2. **Data Preprocessing:**
   - Loads and preprocesses data:
     - Applies geometric mean correction.
     - Normalizes geochemical covariates.
     - Encodes spatio-temporal indicators (biome, province, quarter).

3. **Holdout Mask Creation:**
   - Splits the dataset into training and validation using `h_prop`.

4. **Model Fitting:**
   - Selects the appropriate Stan model based on `mtype`.
   - Fits the model using variational inference.

5. **Performance Evaluation:**
   - Computes LLPD for holdout data using posterior mean estimates.
   - Aggregates results across different initializations and component exclusions.

6. **Output Saving:**
   - Saves posterior samples, LLPD, and diagnostics to designated directories:
     - `../results/component/models/`
     - `../results/component/diagnostics/`

---

### **Outputs**

- **Posterior Results:**
  - File: `"{uid}_{mtype}_{m_seed}_model_nb.pkl"`
  - Contains variational posterior samples from the fitted model.

- **Evaluation Metrics:**
  - File: `"{uid}_{mtype}_{m_seed}_model_nb_cvtest.pkl"`
  - Includes:
    - Holdout mask.
    - LLPD for the holdout set.
    - Predicted values (`mu_sample`) for validation.


### **Error Handling**

- The script handles potential errors during model fitting:
  - If an error occurs (e.g., `ZeroDivisionError`), it saves placeholder results to avoid crashing.

