# DT40Pulling
This repository contains code used for chromosome modeling and analysis in "Structural Integrity and Relaxation Dynamics of Axially Stressed Chromosomes".  We simulate DT40 chromosome 7 using the Minimal Chromatin Model (MiChroM) under a variety of conditions, including the application of external strains and stresses.

openmichrolib.py: This is the library used to carry out molecular dynamics simulations of whole chromosomes.  Input files containing model parameters for the interphase and mitotic chromosomes are found in the folders "Input0m" and "Input15m".

michromanalysislib.py: This library contains functions used to analyze trajectories to calculate simulated contact maps and the properties of renormalized chains.

Remaining folders contain all files necessary to recreate an experiment.  Experiments are summarized here:

OpenSim- Plain simulation of the Minimal Chromatin Model with no additional constraints.

NativeSim- Simulation of chromosome 7 with "pin" and "slide" potentials which constrain the left telomere to the origin and the right telomere to the x-axis, but allow the right telomere to slide along the x-axis

CFSim- Simulation of chromosome 7 with "pin" and "slide" potentials, as well as constant pulling forces (linear potential) between the two telomeres.  Force is turned on and then later released to study the chromosome's dynamical relaxation.

CDSim- Simulation of chromosome 7 with telomeres pinned to specific locations deparated by varying distances.

Within these folders, ".slurm" files were used to submit jobs to a batch scheduler.  One slurm file initiates the simulation, and another slurm file initiates the analysis of calculated trajectories.

After running the analysis scripts for each generated trajectory, move all ".txt" and ".dat" outputs into a folder titled "PlotData."  The "Vis_Plots...ipynb" jupyter notebooks may then be used to plot the calculated values.
