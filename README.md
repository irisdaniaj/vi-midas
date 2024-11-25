# VI-MIDAS: Variational inference for microbiome survey data

This repository contains code and documentation for reproducing the results of [VI-MIDAS](https://github.com/amishra-stats/vi-midas/tree/main). The project aims to use variational inference to jointly model microbial abundance data and environmental factors as well as species-species interaction. This guide will help you set up the environment, reproduce the results, and adapt the code to your data.

---
## Setup 

Clone the repository 
```
git clone https://github.com/irisdaniaj/vi-midas.git
cd vi-midas
```
Create the environment 

```
conda create --name vi-midas python=3.7
conda activate vi-midas
pip install -r requirements.txt
```
---
## Reproducing results 

The next steps will reproduce the results of [VI-MIDAS](https://github.com/amishra-stats/vi-midas/tree/main)

```
cd src
python generate_csv.py
```
This script generates 50 random combinations of hyperparameters (`λ`, `ϑ`, and `k`) within specified ranges using random sampling and saves them to a CSV file named `hyperparams.csv`. It ensures reproducibility by setting a fixed random seed and is useful for preparing a hyperparameter tuning experiment.

```
sbatch submit_all_batches 
```
First, run the `submit_all_batches` script to submit multiple Slurm job arrays, each handling a subset of hyperparameter combinations. This script divides the tasks into smaller batches and assigns different random seeds to each, ensuring a thorough search. After submission, the `mem_hyperparameter_tuning` script is executed for each task, running the hyperparameter tuning process with the specified parameters. You may need to adjust specific cluster-specific configuration in `mem_hyperparameter_tuning` (e.g., `#SBATCH` options). The results will be saved in `results/hyperparameter`. 

---
## Documentation 

For detailed information about the scripts, their functionality, and usage, please refer to the accompanying documentation in `documentation` 

---
## Hardware requirements 

The experiments were conducted on a high-performance computing cluster with 1 node per job, each node equipped with 76 CPU cores, 512 GB RAM, and 8 NVIDIA Tesla V100 GPUs (16 GB HBM2 memory each). Jobs utilized 1 GPU, 2 CPU cores, and 10 GB of memory. Reproducing these results requires similar hardware and resource configurations to ensure comparable performance and outcomes.

---
## Queries
Please contact authors and creators for any queries related to using the analysis 


-   Aditya Mishra: [mailto](mailto:amishra@flatironinstitute.org)
-   Christian Mueller: [mailto](mailto:cmueller@flatironinstitute.org)
