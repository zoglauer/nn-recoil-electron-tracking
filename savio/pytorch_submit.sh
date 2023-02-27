#!/bin/bash

# Remember:
# Submit via: sbatch ...
# Check via: squeue -u $USER

#SBATCH -J Python

#SBATCH --account=fc_cosi
#SBATCH --partition=savio2_gpu
#SBATCH --qos=savio_normal

#SBATCH -t 24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

#SBATCH --signal=2@60

# --> CHANGE TO YOUR EMAIL

##SBATCH --mail-user=XYZ@berkeley.edu



##SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting analysis on host ${HOSTNAME} with job ID ${SLURM_JOB_ID}..."

echo "Loading modules..."
module purge
module load ml/torch/torch7 python/3.7 

echo "Starting execution..."

# --> ADAPT THE FILENAME
source activate torch_env_2
pip install --user pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install --user torch-geometric
cd interaction_network

python3 train.py

echo "Waiting for all processes to end..."
wait
