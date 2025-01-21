### Overview

The `component_submission.py` script automates the submission and execution of component contribution analysis for a generative model. 
It evaluates the impact of excluding different components by dynamically generating and running commands to fit the models with specified hyperparameters.

---

### **Purpose**
1. Automate the execution of the `component_contribution_fit.py` script for multiple component-excluded models.
2. Evaluate model performance for a specific set of hyperparameters (`λ`, `θ`, `k`).
3. Perform multiple initializations (`n_repeats`) for each component-excluded model (`mtype`).

---

### **Command-Line Arguments**

The script accepts the following arguments:

| Argument        | Type   | Description                                            |
|-----------------|--------|--------------------------------------------------------|
| `--lambda`      | float  | Regularization parameter for variance (λ).             |
| `--theta`       | float  | Regularization parameter for mean (ϑ).                |
| `--k`           | int    | Latent rank (dimensionality of the latent variable space). |

**Example Usage:**
```bash
python component_submission.py --lambda 0.05 --theta 0.1 --k 30
```

---

### **Workflow**

1. **Setup:**
   - **Directory Creation:**
     Ensures that the results directory (`../results/component/models/`) exists:
     ```python
     os.makedirs(model_dir, exist_ok=True)
     ```

2. **Command Generation:**
   - For each model type (`mtype = 0–5`) and initialization seed (`n_repeats = 20`), the script generates commands to run `component_contribution_fit.py`.
   - **Command Template:**
     ```bash
     python component_contribution_fit.py <k> <sp_mean> <sp_var> <h_prop> <nsample_0> <sid_current> <mtype> <uid>
     ```
   - **Example Command:**
     ```bash
     python component_contribution_fit.py 30 0.1 0.05 0.1 200 1230101 0 123
     ```

3. **Output File Check:**
   - Before generating a command, the script checks if the corresponding output file already exists:
     ```python
     model_path = os.path.join(model_dir, f"{uid}_{mtype}_{seed}_model_nb.pkl")
     ```
   - If the file exists, the command is skipped to avoid redundant computation.

4. **Parallel Execution:**
   - Uses `ProcessPoolExecutor` to execute commands in parallel, with up to `n_max_run` (default: 2) processes running simultaneously:
     ```python
     with ProcessPoolExecutor(max_workers=n_max_run) as executor:
         results = list(executor.map(run_command, commands))
     ```

5. **Output Logging:**
   - Captures and prints the output and errors for each executed command:
     ```python
     print(f"Command: {result['command']}")
     print(f"Output: {result['stdout']}")
     if result['stderr']:
         print(f"Error: {result['stderr']}")
     ```
---

### **Parameters for the Experiment**

| Parameter       | Default Value | Description                                                                 |
|-----------------|---------------|-----------------------------------------------------------------------------|
| `n_max_run`     | 2             | Number of parallel processes to run.                                       |
| `model_dir`     | `../results/component/models/` | Directory for storing model outputs.                                    |
| `h_prop`        | 0.1           | Holdout proportion for validation data.                                    |
| `nsample_0`     | 200           | Number of posterior samples to generate.                                   |
| `mtype_values`  | `[0, 1, 2, 3, 4, 5]` | Model types to evaluate (full and component-excluded models).          |
| `uid`           | 123           | Fixed simulation setting identifier.                                       |
| `n_repeats`     | 20            | Number of evaluations per model type (initializations).                    |

---

### **Outputs**

1. **Generated Files:**
   - Output files are saved in the `../results/component/models/` directory.
   - File naming convention:
     ```text
     {uid}_{mtype}_{seed}_model_nb.pkl
     ```

2. **Example Outputs:**
   - For `uid=123`, `mtype=0`, and `seed=1`:
     - `123_0_1_model_nb.pkl`
   - For `uid=123`, `mtype=5`, and `seed=20`:
     - `123_5_20_model_nb.pkl`

---

### **Error Handling**

- **File Exists:**  
  Skips commands if the corresponding output file already exists.
- **Command Execution Errors:**  
  Logs errors from `subprocess.run()` for debugging:
  ```python
  return {'command': command, 'stdout': '', 'stderr': f"Error: {str(e)}"}
  ```
