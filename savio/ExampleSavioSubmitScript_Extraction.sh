#!/bin/bash

# Remember:
# Submit via: sbatch ...

#SBATCH -J Sim

#SBATCH --account=fc_cosi
#SBATCH --partition=savio2_gpu
#SBATCH --qos=savio_normal

#SBATCH --chdir=/global/scratch/users/zoglauer/Sims/EnergyLossEstimate

#SBATCH -t 72:00:00

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --signal=2@60

##SBATCH --mail-user=XYZ@berkeley.edu
##SBATCH --mail-type=ALL


echo "Starting submit on host ${HOST}..."

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

. /global/home/groups/fc_cosi/MEGAlib/bin/source-megalib.sh

echo "Starting execution..."

# --> ADAPT THE FILENAME
python3 -u /global/home/users/rbohra/RecoilElectronTracking/EventExtractor.py -m 1000000 -f RecoilElectrons.1000k.data

echo "Waiting for all processes to end..."
wait