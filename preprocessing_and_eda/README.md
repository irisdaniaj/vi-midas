# Preprocessing and Exploratory Analysis

This folder contains all preprocessing, imputation, and exploratory analysis scripts used for data cleaning and exploration prior to modeling. It is designed to be standalone and reproducible using Python and R.

---

## Folder Structure

```
preprocessing_and_eda/
│
├── data/              # Input data files - all required data files (e.g., .csv)
├── results/           # Outputs: plots, tables, results
├── src/               # All analysis scripts (R and Python)
│   ├── eda_numeric.R
│   ├── richness_evenness.R
│   ├── heatmap.R
│   ├── world_maps.R
│   ├── nmds.R
│   ├── motu_filtered.py
│   └── mice.py
│
├── run.sh             # for mice.py
├── environment.yml    # Conda environment for Python dependencies
└── README.md          # This file
```
# Note: Output files are saved to `results/` or `data/`, or new subfolders are created as needed.

###  Setup Instructions
### Conda Environment (for Python scripts)

Ensure Conda is installed. Then from terminal run:
```bash
cd preprocessing_and_eda
conda env create -f environment.yml
conda activate myenv  # or replace 'myenv' with your environment name


### How to run the Python scripts from terminal : 
cd preprocessing_and_eda
conda activate myenv 
# Run Python scripts
python src/mice.py
python src/motu_filtered.py


### How to run the R scripts  : 
# Step 1. Open them in R Studio
# Step 2. set you working directory : #setwd("/PUT/YOUR/PROJECT/PATH/HERE")
# Step 3. All required packages will install and load automatically when the script is run

## Scripts Overview

- `eda_numeric.R`: Exploratory data analysis on numeric variables
- `richness_evenness.R`: Calculates microbial richness and evenness
- `heatmap.R`: creates a clustered heatmap of filtered mOTU data across samples
- `world_maps.R`: Plots sampling locations on world maps using metadata
- `nmds.R`: Performs NMDS analysis and PERMANOVA
- `motu_filtered.py`: Filters OTU data for downstream use
- `mice.py`: Imputes missing values using MICE - best per variable strategy

---

##   Optional: Run MICE Imputation via `run.sh`
A helper shell script `run.sh` is included to run the MICE imputation script automatically.

### Contents of `run.sh`:
```bash
#!/bin/bash

# Exit on any error
set -e

# Load conda
source /opt/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate myenv

# Run your script
python src/mice.py
