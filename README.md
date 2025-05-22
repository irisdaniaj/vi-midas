# VI-MIDAS: Variational inference for microbiome survey data

This repository contains code and documentation for reproducing the results of [VI-MIDAS](https://github.com/amishra-stats/vi-midas/tree/main). The project aims to use variational inference to jointly model microbial abundance data and environmental factors as well as species-species interaction. This guide will help you set up the environment, reproduce the results, and adapt the code to your data.

---
## Contribution 

### Reproduction of Original VI-MIDAS Results
We successfully reproduced the results of the original VI-MIDAS model using the same dataset and model configurations described by [Mishra et al. (2024)](https://www.biorxiv.org/content/10.1101/2024.03.18.585474v1). This involved modeling overdispersed microbial abundance data using a Negative Binomial likelihood and integrating both direct environmental covariates and latent space components for spatiotemporal factors and taxon-taxon interactions. 

### Exploratory Data Analysis 
Before modeling, we conducted an extensive exploratory data analysis that encompassed both the original VI-MIDAS dataset and our newly assembled microbiome dataset. The new dataset expanded the geographic and ecological coverage, particularly with additional samples from polar biomes, and incorporated a richer set of covariates, including satellite-derived environmental features.
We examined differences in environmental covariates, such as temperature, salinity, and oxygen—between polar and non-polar samples. We visualized mOTU abundance profiles and observed ecological stratification across biomes and depth layers. 

### Application of VI-MIDAS to a New Marine Microbiome Dataset
Building on the original methodology, we applied the VI-MIDAS framework to a newly assembled marine microbiome dataset. Both the direct coupling model (which links environmental covariates directly to microbial abundances) and the no direct coupling variant (which projects all covariates into a shared latent space) were evaluated. 

### Development and Evaluation of a Novel Modeling Variant
We implemented a novel no direct coupling variant of VI-MIDAS (detailed in Section 3.1.3 of the report), which differs from the original model by projecting environmental covariates—along with spatiotemporal and interaction-based factors—into a shared latent space. This unified representation allowed for a more flexible and holistic interpretation of ecological patterns. We applied this model to both the original and the new datasets. 

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

In the `config_mode.txt` file please write in the first line which data you want to analyze "original" or "new" and then select then which model to use "1" direct coupling, "2" no direct coupling.

The next steps will reproduce the results of [VI-MIDAS](https://github.com/amishra-stats/vi-midas/tree/main)

```
cd src
python generate_csv.py
```
This script generates 50 random combinations of hyperparameters (`λ`, `ϑ`, and `k`) within specified ranges using random sampling and saves them to a CSV file named `hyperparams.csv`. It ensures reproducibility by setting a fixed random seed.

### Exploratory Data Analysis 

To reproduce our EDA results please refer to the `preprocessing_and_data` folder.  

### Hyperparameter tuning

```
sbatch submit_run_hyperparameter
```
The `submit_run hyperparameter` script submits as a batch job, which automates the execution of `hyperparameter_tuning_fit.py` for all hyperparameter combinations specified in `hyperparams.csv`. The results will be saved in `results/hyperparameter`.  To analyze the results of `hyperparameter_tuning_fit.py` and choose the best hyperparameter combination in terms of out of sample log-likehood predictive density now we run the `hyperparameter_tuning_analysis.ipyb` notebook. The generated plots are saved in `results/plot`. 

### Component contribution

Now that the best hyperparameter combination has been choosen we will analyze the contribution of each component(e.g., geochemical, spatio-temporal, interaction) in the generative model by dropping each single component for the given hyperparameter combination. To do so, run 
```
sbatch submit_run_component 
```
This submits `run_component.py` as a batch job, which automates the execution of `component_contribution_fit.py` for the selected hyperparameter combination. To analyze the results of `component_contribution_fit.py` and identify how much each component contributes to the overall model performance run the `component_contribution_analysis.ipyb` notebook. The generated plots are saved in `results/plot`. 

### Model sensitivity

Since the parameters estimate are sensitive to the choice of their initial estimates. We further evaluate the choosen hyperparameter set for 50 random initialization and then select the best model out of it.
```
sbatch submit_run_component 
```
This submits `run_sensitivity.py` as a batch job, which automates the execution of `model_sensitivity_fit.py` for the selected hyperparameter combination. To analyze the results of `model_sensitivity_fit.py` and identify the best model run the `model_sensitivity_analysis.ipyb` notebook. The generated plots are saved in `results/plot`. 

### Analysis 

Before the analysis select which dataset and which methodology result you want to analysize in the  `notebooks/config_mode.txt` file. 
Now we can analyze the best model parameter estimates. \
`model_analysis1.ipynb`: Analyzes model parameter estimates. \
`model_analyis2.ipynb`: Evaluates model validity using different performance metrics. \
`model_analysis3.ipynb`: Investigates the contribution of individual components in the MEM. 

### Inference 

Now we can run inference. \
`model_inference1.ipynb`: Infers the effects of geochemical factors and spatio-temporal components. \
`model_inference2.ipynb`: Identify species similarity using cosine distance on the latent vector and detect positive interactions among species. \
`model_inference3.ipynb`: Similar to the previous notebook but with different visualization. 

---
## Documentation 

For detailed information about the scripts, their functionality, and usage, please refer to the accompanying documentation in `documentation` 

---
## Hardware requirements 

The experiments were conducted on a high-performance computing cluster with 1 node per job, each node equipped with 76 CPU cores, 512 GB RAM, and 8 NVIDIA Tesla V100 GPUs (16 GB HBM2 memory each). Jobs utilized 1 GPU, 2 CPU cores, and 10 GB of memory. Reproducing these results requires similar hardware and resource configurations to ensure comparable performance and outcomes.

---

