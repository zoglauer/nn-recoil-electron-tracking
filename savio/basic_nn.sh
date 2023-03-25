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
module load python/3.7 #ml/torch/torch7
source activate torch_env_3
echo completed torch install

echo "Starting execution..."
cd torch_basic_network
model_iter=modelv1
mkdir $model_iter
python3 train.py --lr .0001 --batch-size 4 --epochs 100 --model-iteration $model_iter >> $model_iter.txt

echo "Batch Completed"
wait
