#!/bin/bash

#SBATCH --job-name=RREPNUMNF0m
#SBATCH --account=ctbp-onuchic
#SBATCH --partition=ctbp-onuchic
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --time=13:45:59
#SBATCH --mem-per-cpu=1G
#SBATCH --export=ALL
#SBATCH --constraint="skylake&opath"

module load GCC/8.3.0 CUDA/10.1.168 OpenMPI/3.1.4 h5py/2.9.0

srun ~/anaconda3/bin/python NoForce_0m.py
