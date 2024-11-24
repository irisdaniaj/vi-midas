#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --array=0-49%10
#SBATCH --job-name=hyperparam_tuning
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=2:00:00

# Activate Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vim

# Debugging: Check environment and Python path
echo "Active Conda Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Using Python: $(which python)"
echo "Python Version: $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"

python3 -c "
try:
    import pystan
    print('Pystan is installed.')
    print('Pystan location:', pystan.__file__)
except ModuleNotFoundError as e:
    print('Error: Pystan is not installed. Exception:', e)
"

