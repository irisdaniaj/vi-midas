#!/bin/bash

# Exit on any error
set -e

# Load conda
source /opt/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate myenv

# Run your script
python src/mice.py
