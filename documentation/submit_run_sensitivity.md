# SLURM Batch Script Documentation

## Overview
This SLURM batch script is designed to submit a job array for performing **sensitivity analysis** using the `run_sensitivity.py` script.

## SLURM Job Parameters
You may need to adapt the job parameters to your cluster configurations

## Execution
This script runs the following command for each task in the job array:
```bash
python3 run_sensitivity.py $SLURM_ARRAY_TASK_ID 2>&1
```
Where:
- `$SLURM_ARRAY_TASK_ID` represents the task ID (from 1 to 4), allowing for different sensitivity analysis configurations per job instance.
- `2>&1` redirects standard error to standard output.

## Debugging Information
To assist with debugging, the script prints:
- The SLURM task ID being executed.
- The working directory path.
- The Python path (`which python3`).
- The active Conda environment (`$CONDA_DEFAULT_ENV`).

## Logs
- **Output logs** are saved in:
  ```
  slurm-%A_%a.out
  ```
- **Error logs** are saved in:
  ```
  slurm-%A_%a.err
  ```
Where `%A` is the SLURM job ID and `%a` is the task ID.

## Completion Message
Once execution is complete, the script outputs:
```bash
echo "Job finished with exit code $?"
```
This helps track any exit status for troubleshooting.
