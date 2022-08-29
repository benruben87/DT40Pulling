#!/bin/bash

#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:29:59
#SBATCH --mem-per-cpu=1G
#SBATCH --export=ALL
#SBATCH --job-name=15mNat

module load GCC/8.3.0 CUDA/10.1.168 OpenMPI/3.1.4 h5py/2.9.0

srun ~/anaconda3/bin/python Native-15m.py
