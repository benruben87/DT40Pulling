#!/usr/bin/env python
# coding: utf-8

# # DT40 15m Constant Force Simulation Individual Trajectory Analysis.
# Loads data from files saved during CF simulation and calculates contact maps, coarse-grained bond cosines, and MSD of pull coordinate.

# In[1]:

import michromanalysislib as malib
from openmichrolib import Akroma as openmichrom
from openmichrolib import SerraAngel as trainertool
import time
import numpy as np

from simtk.openmm.app import *
import simtk.openmm as openmm
import simtk.unit as units
from sys import stdout, argv
import numpy as np
from six import string_types
import os
import time
import random
import h5py
from scipy.spatial import distance
import scipy as sp
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt

#Don't show figures, only save:
mpl.use('Agg')

plt.rcParams.update({'font.size': 16, 'figure.figsize': [6.0, 5.0]})


# ## Simulation Parameters (To Be Changed Before Running on NOTS):

# In[17]:


repnum = REPNUM # VARIABLE: REPNUM.  Set with sed command
forcejump = FJUMP # VARIABLE: FJUMP interval between forces
forcenum = FORCENUM # VARIABLE: FORCENUM. Current index of force applied
timestep=.01

force = forcenum*forcejump

platform = 'CPU'
blockSize = 100 #Pull Coordinate is recorded every block, whicih is every 10 steps
numBlocks = 6*10**4
blocksPerFrame = 10 #Positions of all beads are recorded every frame, which is every 10 blocks (1000 steps) in this case.

platform_release = 'CPU'
blockSize_release = 100
numBlocks_release = 2*10**4
blocksPerFrame_release = 10

# # Analysis of cf simulation

# ## Contact Maps.
#Number of frames to skip when calculating mean energy components, contact maps, and bond cosines.
#Should be long enough to allow chromosome to completely relax after application of constant force.
eqFrames = 1500
n_arr = [10, 20, 50, 100, 150, 200, 300]
max_tau = 20000

mu            = 1.51       # parameter for contact function f
r_cut         = 2.12       # parameter for contact function f
chronum = 7
res  = 50000

trajFile = 'output_files/CF_'+str(forcenum)+'_traj_0.cndb'

#Contace Maps
P = malib.calcContProbs(trajFile, mu, r_cut, eqFrames, chronum)
np.savetxt('output_files/Prob_'+str(forcenum)+'.dat', P)
del P

# ## Bending Analysis (Smoothed Skeletons)

Len_Cos_Rg_Dict = malib.calc_lens_cos_rg_traj(trajFile, n_arr = n_arr, eqFrames = eqFrames)

# Store data (serialize)
f = open('output_files/CosDict_'+ str(forcenum) +'.txt',"w")
f.write( str(Len_Cos_Rg_Dict) )
f.close()
del Len_Cos_Rg_Dict

# ## Calculate Renormalized Bond Length as a function of renormalization scale b
BondLens = malib.calc_renormalized_bondlens_traj(trajFile, skipnum=1, eqFrames = eqFrames)
np.savetxt('output_files/BondLens_'+ str(forcenum) +'.txt', BondLens)
del BondLens

# ##Determine Chromosome Radial Distribution:
radHistData = malib.calc_axis_radial_dist_traj(trajFile, skipnum=1, axes = [0,1,1])

# Store histogram data: 
#first line is bin edges 
#Second line is probability density.
#Leaving Code in general form for Analysis script which will combine histograms for the many forces.
f = open('output_files/RadHist_'+ str(forcenum) +'.txt',"w")

for row in range(len(radHistData)-1):
    for n in radHistData[row]:
        f.write(str(n) + '    ')
    f.write('\n')

for n in radHistData[len(radHistData)-1]:
    f.write(str(n)+ '    ')

f.close()
del radHistData
# ## MSD of reaction coordinate:

dt = timestep*blockSize #time step between coordinate measurements.
max_Steps = int(max_tau/dt)
eqBlocks = eqFrames*blocksPerFrame

curfile = 'output_files/CF_'+str(forcenum)+'_Pull_Coord.txt'
Coord_Data = np.genfromtxt(curfile,dtype='f8', delimiter=' ')
pcoord = Coord_Data[eqBlocks:,1]

MSDs = []
for steps in range(max_Steps):
    MSDs.append(np.mean(np.square(np.subtract(pcoord[eqBlocks:len(pcoord)-steps], pcoord[eqBlocks+steps:]))))
    
tau = np.linspace(0,dt*len(MSDs), len(MSDs))
MSD_Data = np.vstack([tau, MSDs]).T
np.savetxt('output_files/CF_'+str(forcenum)+'_MSD.txt', MSD_Data)
