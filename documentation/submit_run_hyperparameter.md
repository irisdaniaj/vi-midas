# SLURM Batch Script Documentation

## Overview
This SLURM batch script is used to submit a job array for running the `run_hyperparameter.py` script in a high-performance computing environment. 

## SLURM Job Parameters
 You may need to adapt the job parameters to fit your clusters. 

## Execution
This script runs the following command for each task in the job array:
```bash
python3 run_hyperparameter.py $SLURM_ARRAY_TASK_ID 2>&1
```
Where:
- `$SLURM_ARRAY_TASK_ID` represents the task ID (from 1 to 5), allowing for different hyperparameter settings per job instance.
- `2>&1` redirects standard error to standard output.

## Logs
- **Output logs** are saved in:
  ```
  vi-midas/src/log/vimidas-%A_%a.log
  ```
- **Error logs** are saved in:
  ```
  /vi-midas/src/log/vimidas-%A_%a.err
  ```
Where `%A` is the SLURM job ID and `%a` is the task ID.

## Notes
- Modify the job array range (`--array=1-5`) to match the number of hyperparameter settings needed.


