#!/bin/bash -l
#SBATCH --job-name=Test
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=/home/ucabc46/exp/tiny-cnns
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

/home/ucabc46/exp/tiny-cnns/benchmark -b 1 -i 50