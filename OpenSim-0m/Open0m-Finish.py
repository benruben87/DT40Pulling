#!/usr/bin/env python
# coding: utf-8

# # DT40 0m Dry Run Finishing Script.
# Calculates contact map, smoothed skeletons, mean bond smoothed bond lengths, and chromosome radius from single run.

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

# ## Simulation Parameters (To Be Changed Before Running on NOTS):

blockSize = 5000
numBlocks = 50
eqFrames = 5 #Number of frames to skip when calculating mean energy components, contact maps, and bond cosines.
n_arr = [1, 10, 50, 100, 150, 200, 300]

#Don't show figures, only save:
mpl.use('Agg')

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
radHistData = malib.calc_axis_radial_dist_traj(trajFile, skipnum=1, axes = [1,1,1], eqFrames= eqFrames)#interphase chromosome is treated as spherical.
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