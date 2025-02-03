#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/dss/dsshome1/09/ra64nef3/vi-midas/src/log/vimidas-%A_%a.log
#SBATCH --error=/dss/dsshome1/09/ra64nef3/vi-midas/src/log/vimidas-%A_%a.err
#SBATCH -D ./
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --mail-type=NONE
#SBATCH --mail-user=iris.jimenez@campus.lmu.de

echo "Test SLURM submission"
