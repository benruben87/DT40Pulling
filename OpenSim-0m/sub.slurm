#!/bin/bash

#SBATCH --job-name=Open0m
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=9:59:59
#SBATCH --mem-per-cpu=2G
#SBATCH --export=ALL


module load GCC/8.3.0 CUDA/10.1.168 OpenMPI/3.1.4 h5py/2.9.0

srun ~/anaconda3/bin/python Open0m.py
#srun ~/anaconda3/bin/python ../../NDB-Converters-master cndb2ndb.py -f chr_10_pull_0.cndb -n chr_10_pull_0
