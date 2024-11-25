# Hyperparameter Tuning - Batch Submission and Execution

## Overview

Two bash scripts used to perform hyperparameter tuning across a set of hyperparameter combinations. The first script, `submit_all_batches`, submits multiple batches of jobs to the Slurm scheduler. The second script, `mem_hyperparameter_tuning`, runs the actual hyperparameter tuning by executing a Python script (`hyperparameter_tuning_fit.py`) with a specific set of hyperparameters. The two scripts work together to efficiently distribute and run hyperparameter tuning tasks in parallel.

---

## Script 1: `submit_all_batches.sh`

### Purpose:
The `submit_all_batches` script is responsible for submitting multiple job batches to the Slurm scheduler. Each batch consists of a subset of tasks with different random seeds for hyperparameter tuning. The script loops through batches and submits jobs with different `OFFSET` values, ensuring that the tasks are distributed evenly across multiple Slurm job arrays.

### Key Features:
- **GPU Request**: Requests 2 GPUs per job.
- **Parallel Job Submission**: Submits jobs in batches using `sbatch --array=0-9%10`, which runs up to 10 jobs in parallel.
- **Offset Management**: The `OFFSET` value is incremented for each batch, allowing different sets of hyperparameter combinations to be processed with various seeds.
- **Job Array Submission**: The script uses `sbatch` to submit each batch of jobs and includes a 20-second delay (`sleep 20`) to avoid overwhelming the scheduler.

### How it Works:
1. The script starts a loop from `0..24` to submit 25 batches of jobs.
2. For each iteration, it calculates a unique `OFFSET` value for each batch and submits a job array using `sbatch`.
3. The script exports the `OFFSET` variable to each submitted job, ensuring that the correct hyperparameters are used for each batch.

## Script 2 mem_hyperparameter_tuning

The `mem_hyperparameter_tuning` script is used for running hyperparameter tuning tasks on a cluster using the Slurm job scheduler. It reads hyperparameter combinations from a CSV file, then executes a Python script (`hyperparameter_tuning_fit.py`) for each task, tuning the model with the selected parameters. The script is designed to be used in conjunction with the `submit_all_batches` script, which submits job arrays for parallel execution.

## Key Features

- **Slurm Job Array**: Uses `#SBATCH --array=0-49%10 ` to run up to 50 tasks in parallel. Each task corresponds to a different hyperparameter combination.
- **Hyperparameter Parsing**: Extracts hyperparameters from a CSV file (`hyperparams.csv`), where each row contains the values for λ, ϑ, and k.
- **Logging**: Creates logs for each task, saving output and error logs in `../results/hyperparameter/logs/` and task-specific logs in `../results/hyperparameter/logfile/`.
- **Environment Setup**: Activates a Conda environment (`vim`) to ensure the correct dependencies are available for running the Python script.
- **Task Distribution**: Each task is assigned a unique set of hyperparameters and a random seed based on the job array index and an `OFFSET` value passed by the `submit_all_batches` script.

## How It Works

1. **Task Variables**:
   - **Task ID**: Calculated from the Slurm array task ID and an `OFFSET` value to determine which hyperparameter combination to use.
   - **Seed**: Each task is assigned a seed value (0-4) to run the same combination of hyperparameters with different random seeds, ensuring variability in the results.
   - **Row ID**: Used to map the Slurm task to a specific row in the hyperparameters CSV file.

2. **CSV Parsing**: The script reads the hyperparameters (λ, ϑ, k) for the current task from the `hyperparams.csv` file. 

3. **Parameter Validation**: The script checks if the hyperparameters (λ, ϑ, k) are correctly parsed from the CSV. If any values are missing or invalid, it exits with an error message.

4. **Python Script Execution**: For each task, the script runs the Python script `hyperparameter_tuning_fit.py` with the selected hyperparameters.

5. **Logging**: Each task's logs are saved to `../results/hyperparameter/logfile/task_${TASK_ID}.log`, while Slurm output and error logs are saved to `../results/hyperparameter/logs/`.


### Running the Script
This script is typically executed as part of a larger job submission process (through `submit_all_batches`). However, it can also be run directly after setting the appropriate `OFFSET` environment variable.


