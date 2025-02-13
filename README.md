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
conda create --name vim python=3.7
conda activate vim
pip install -r requirements.txt
```
---
## Reproducing results 

The next steps will reproduce the results of [VI-MIDAS](https://github.com/amishra-stats/vi-midas/tree/main)

```
cd src
python generate_csv.py
```
This script generates 50 random combinations of hyperparameters (`λ`, `ϑ`, and `k`) within specified ranges using random sampling and saves them to a CSV file named `hyperparams.csv`. It ensures reproducibility by setting a fixed random seed.

### Hyperparameter tuning

```
sbatch submit_run_hyperparameter
```
The `submit_run hyperparameter` script submits `_hyperunrparameter.py` as a batch job using SLURM, this script automates the execution of `hyperparameter_tuning_fit.py` for all hyperparameter combinations specified in `hyperparams.csv`. The results will be saved in `results/hyperparameter`.  To analyze the results of `hyperparameter_tuning_fit.py` and choose the best hyperparameter combination in terms of out of sample log-likehood predictive density now we run the `hyperparameter_tuning_analysis.ipyb` notebook. The generated plots are saved in `results/plot`. 

### Component contribution

Now that the best hyperparameter combination has been choosen we will analyze the contribution of each component(e.g., geochemical, spatio-temporal, interaction) in the generative model by dropping each single component for the given hyperparameter combination. To do so, run 
```
sbatch submit_run_component 
```
This submits `run_component.py` as a batch job, which automates the execution of `component_contribution_fit.py` for the selected hyperparameter combination. To analyze the results of `component_contribution_fit.py` and identify how much each component contributes to the overall model performance run the `component_contribution_analysis.ipyb` notebook. The generated plots are saved in `results/plot`. 

### Model sensitivity
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
