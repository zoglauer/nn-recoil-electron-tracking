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
#================================================================================

apptainer run container.sif
pip3 install --user torch
pip3 install --user torch_geometric
pip3 install --user numpy
pip3 install --user torch-sparse==0.6.13 -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
python3 interaction_network/train.py

#================================================================================
echo "Batch Completed"
wait
