### Overview

This SLURM batch script submits a job to execute the `component_submission.py` Python script on a high-performance computing (HPC) cluster.
It accepts hyperparameters as command-line arguments and ensures the required resources are allocated for the job.

---

### **Purpose**

The script:
1. Allocates computational resources (e.g., GPU, memory, runtime) via SLURM.
2. Activates a Python environment required for the analysis.
3. Passes user-specified hyperparameters (`λ`, `ϑ`, `k`) to the `component_submission.py` script.
4. Logs the output and errors for debugging and progress tracking.

---

### **SLURM Parameters**

1. **Partition (`#SBATCH -p mcml-dgx-a100-40x8`):**
   - Specifies the GPU-enabled partition for running the job.

2. **Quality of Service (`#SBATCH --qos=mcml`):**
   - Configures the priority and resource access for the job.

3. **Tasks and GPUs:**
   - `#SBATCH --ntasks=1`: Requests a single task.
   - `#SBATCH --gres=gpu:1`: Allocates one GPU.

4. **Runtime (`#SBATCH -t 47:00:00`):**
   - Sets a maximum runtime of 47 hours for the job.

---

### **Script Functionality**

1. **Command-Line Argument Parsing:**
   - The script reads the following hyperparameters:
     | Argument        | Description                                       |
     |-----------------|---------------------------------------------------|
     | `-l` or `--lambda` | Regularization parameter for variance (λ).         |
     | `-theta`        | Regularization parameter for mean (ϑ).            |
     | `-k`            | Latent rank (k), which defines the latent space.  |
   - Example:
     ```bash
     sbatch submit_component.sh -l 0.05 -theta 0.1 -k 30
     ```

   - If any argument is missing, the script exits with an error message:
     ```bash
     echo "Usage: sbatch submit_component.sh -l <lambda> -theta <theta> -k <k>"
     ```

---

### **Outputs**

1. **Logs:**
   - All standard output and errors are redirected to `logs/submission.log`.

2. **Model Results:**
   - The Python script (`component_submission.py`) generates results in the `../results/component/models/` directory.

---

### **Error Handling**

1. **Missing Arguments:**
   - If any required argument (`λ`, `ϑ`, `k`) is missing, the script exits with an error:
     ```bash
     echo "Usage: sbatch submit_component.sh -l <lambda> -theta <theta> -k <k>"
     ```

2. **Python Environment Activation:**
   - Ensure the Conda environment (`vim`) exists and is correctly configured.

3. **Job Failures:**
   - Review the SLURM output or `logs/submission.log` for debugging.

---

### **Customizations**

1. **Adjust SLURM Parameters:**
   - Modify the partition, runtime, or GPU allocation as needed:
     ```bash
     #SBATCH -p <desired_partition>
     #SBATCH --gres=gpu:<number_of_gpus>
     ```

2. **Change the Python Environment:**
   - Replace `vim` with the name of your Conda environment.

3. **Redirect Logs:**
   - Specify a custom log file:
     ```bash
     python3 component_submission.py --lambda "$LAMBDA" --theta "$THETA" --k "$K" > logs/custom_log.log 2>&1
     ```
