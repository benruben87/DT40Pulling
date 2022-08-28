#!/bin/bash -l

#SBATCH --job-name=CF0mFinish
#SBATCH --partition=ctbp-onuchic
#SBATCH --account=ctbp-onuchic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH --time=23:30:00

export OMP_NUM_THREADS=1

module load GCC/8.3.0 CUDA/10.1.168 OpenMPI/3.1.4 h5py/2.9.0 GCC/5.4.0 matplotlib/3.2.1 

srun ~/anaconda3/bin/python  CF_FORCENUM_Finish_0m.py
