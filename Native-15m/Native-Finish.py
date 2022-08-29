#!/usr/bin/env python
# coding: utf-8


import michromanalysislib as malib
from openmichrolib import Akroma as openmichrom
from openmichrolib import SerraAngel as trainertool
import time
import numpy as np
import h5py
from   scipy.spatial  import distance
import sys
from numpy.random import random as rand
from numpy.linalg import norm
from itertools import islice, product, combinations
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from scipy.ndimage.filters import uniform_filter1d

#Don't show figures, only save:
mpl.use('Agg')

platform = 'cuda'
blockSize = 10 #Pull Coordinate is recorded every block
numBlocks = 5*10**5
blocksPerFrame = 100 #Positions of all beads are recorded every Frame.
max_tau = 20000
eqFrames = 10 #Number of frames to skip when calculating bond cosines.
n_arr = [1, 10, 50, 100, 150, 200, 300]
timestep = .01
eqBlocks = eqFrames*blocksPerFrame

curfile = 'output_files/Pull_Coord.txt'
Coord_Data = np.genfromtxt(curfile,dtype='f8', delimiter=' ')
pcoord = Coord_Data[eqBlocks:,1]

# ## MSD of reaction coordinate:
dt = timestep*blockSize #time step between coordinate measurements.
max_Steps = int(max_tau/dt)
eqBlocks = eqFrames*blocksPerFrame
MSDs = []
for steps in range(max_Steps):
    MSDs.append(np.mean(np.square(np.subtract(pcoord[eqBlocks:len(pcoord)-steps], pcoord[eqBlocks+steps:]))))
    print(steps)
tau = np.linspace(0,dt*len(MSDs), len(MSDs))
MSD_Data = np.vstack([tau, MSDs]).T
np.savetxt('output_files/Native_MSD.txt', MSD_Data)

# # Analysis
# 
# ## Contact Maps

start         = time.time()

mu            = 1.51       # parameter for contact function f
r_cut         = 2.12       # parameter for contact function f

res  = 50000
trajFile = 'output_files/traj_0.cndb'
chronum=7

#Contact Map
P = malib.calcContProbs(trajFile, mu, r_cut, eqFrames, chronum)
np.savetxt('output_files/Prob{:}_{:}.dat'.format(chronum, chronum), P)

# ## Bending Analysis (Smoothed Skeletons)
Len_Cos_Rg_Dict = malib.calc_lens_cos_rg_traj(trajFile, n_arr = n_arr, eqFrames = eqFrames)
# Store data (serialize)
f = open("output_files/CosDict.txt","w")
f.write(str(Len_Cos_Rg_Dict))
f.close()

# ## Calculate Renormalized Bond Length as a function of renormalization scale b
BondLens = malib.calc_renormalized_bondlens_traj(trajFile, skipnum=1, eqFrames = eqFrames)
np.savetxt("output_files/BondLens.txt", BondLens)

# # Calculate Radial Distribution
radHistData = malib.calc_axis_radial_dist_traj(trajFile, skipnum=1, axes = [0,1,1], eqFrames= eqFrames)#prometaphase chromosome is treated as cylindrical.
# Store histogram data:
#first line is bin edges 
#Second line is probability density.
#Leaving Code in general form for later when there will be a different histogram for each force or distance.
f = open("output_files/RadHist.txt","w")

for row in range(len(radHistData)-1):
    for n in radHistData[row]:
        f.write(str(n) + '    ')
    f.write('\n')

for n in radHistData[len(radHistData)-1]:
    f.write(str(n)+ '    ')

f.close()