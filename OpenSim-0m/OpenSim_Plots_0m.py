#!/usr/bin/env python
# coding: utf-8

# # OpenSim Plotting Scripts
# ## Combines data outputs from runs in folders 1,2,.... numReps to plot means over multiple replicas.  Contact maps, bond cosines, and energy components
# ## Saves necessary data to be used for making plots combining various chromosomes later.

import sys
import numpy as np
import scipy as sc
import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from random import randint
from array import *
import fileinput
import itertools
import linecache
from itertools import islice
from scipy.spatial import distance

#Don't show figures, only save:
mpl.use('Agg')

if not os.path.exists('PlotData'):
    os.makedirs('PlotData')
    
if not os.path.exists('Plots'):
    os.makedirs('Plots')


# Variables:
numBeads = 738
numReps = 3
jobName = '0mOpenSim'
enerFName = 'MeanEner.dat'
bondLenFName = 'BondLens.txt'
radHistFName = 'RadHist.txt'
contMapFName = 'Prob7_7.dat'
cosFName = 'CosDict.txt'
Time_Label = '0m'


#Contact Map:
l=numBeads

start = time.time()

rm=np.zeros((l,l))
ctr = 0;
for i in range(1, numReps+1):
    
    curfile = str(i)+'/output_files/'+contMapFName
    rm = rm + np.genfromtxt(curfile,dtype='f8', delimiter=' ')
    ctr+=1

r = rm/ctr
print(i)

print(r.max(),r.min())
sys.stdout.flush()

np.savetxt('PlotData/'+jobName+'-Dist-all.dat',r,fmt="%s")
fig, ax = plt.subplots()
ax.matshow(r,norm=mpl.colors.LogNorm(vmin=0.0001, vmax=r.max()),cmap="Reds")
#fig.show()

fig = plt.gcf()
fig.savefig('Plots/ContMap_'+jobName+'.png', dpi = 300)

sys.stdout.flush()
end = time.time()
elapsed = end - start
print("Ran in %f sec" % elapsed)

# P vs. D from contact map:
Prob = []
for i in range(l):
    ps = [r[j, j+i] for j in range(l-i)]
    Prob.append(np.mean(np.array(ps)))

np.savetxt('PlotData/'+jobName+'-PvD-all.dat',np.matrix([np.array(range(l)), np.array(Prob)]),fmt="%s")
    
fig, ax = plt.subplots()
ax.plot(50*np.array(range(l)),Prob)
ax.set_xlabel('Genomic Distance (kb)')
ax.set_ylabel('Mean Contact Probability')

fig = plt.gcf()
fig.savefig('Plots/PvD_'+jobName+'.png')    


# # Energy Components:
AllEnergies = []
EnergyNames = []

for i in range(1, numReps+1):
    curfile = str(i)+'/output_files/'+enerFName
    
    file1 = open(curfile, 'r') 
    Lines = file1.readlines() 
    EnergyNames = Lines[0].strip().split('    ')
    curEnergies = Lines[1].strip().split('    ')
    curEnergies = [float(curener) for curener in curEnergies]
    AllEnergies.append(curEnergies)
    

AllEnergies = np.matrix(AllEnergies)
MeanEnergies = np.array(AllEnergies.mean(0))[0]
StdEnergies = np.array(AllEnergies.std(0))[0]
np.savetxt('PlotData/'+jobName+'-MeanEnergies-AllReps.dat',[EnergyNames, MeanEnergies, StdEnergies],fmt="%s")


x = np.arange(len(EnergyNames))  # the label locations
width = 0.7  # the width of the bars
fig, ax = plt.subplots()
fig.set_figwidth(15)
print(x)
print(MeanEnergies)
print(StdEnergies)

ax.bar(x = x, height = MeanEnergies, width = width, yerr = StdEnergies)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel(r"Energy ($\epsilon$)")
ax.set_title(jobName)
ax.set_xticks(x)
ax.set_xticklabels(EnergyNames)
#fig.show()

fig = plt.gcf()
fig.savefig('Plots/Energies_'+jobName+'.png', dpi = 300)


# # Bond Cosines
# importing the module 
import ast 

AllCosDicts = []

for i in range(1, numReps+1):
    curfile = str(i)+'/output_files/'+cosFName
    # reading the data from the file 
    with open(curfile) as f: 
        data = f.read() 
    print("Data type before reconstruction : ", type(data)) 
    
    #print(data)
    # reconstructing the data as a dictionary 
    d = ast.literal_eval(data) 
    print("Data type after reconstruction : ", type(d)) 
    #print(d)
    AllCosDicts.append(d)
    
MeanCosDict = {}
n_arr = list(AllCosDicts[0].keys())

for n in n_arr:
    MeanCosDict[n] = {}
    Lens = []
    Cosines = []
    RGs = []
    for CosDict in AllCosDicts:
        Lens.append(CosDict[n]['len'])
        Cosines.append(CosDict[n]['cos'])
        RGs.append(CosDict[n]['rg'])
    
    Lens = np.matrix(Lens)
    Cosines = np.matrix(Cosines)
    MeanLens = np.array(Lens.mean(0))[0]
    StdLens = np.array(Lens.std(0))[0]
    MeanCosines = np.array(Cosines.mean(0))[0]
    StdCosines = np.array(Cosines.std(0))[0]
    MeanRG = np.mean(RGs)
    StdRG = np.std(RGs)
    
    MeanCosDict[n]['len']=MeanLens.tolist()
    MeanCosDict[n]['stdlen'] = StdLens.tolist()
    MeanCosDict[n]['cos'] = MeanCosines.tolist()
    MeanCosDict[n]['stdcos'] = StdCosines.tolist()
    MeanCosDict[n]['rg'] = MeanRG
    MeanCosDict[n]['stdrg'] = StdRG
## Save MeanCosDict Here

# Store data (serialize)
f = open("PlotData/MeanCosDict.txt","w")
f.write( str(MeanCosDict) )
f.close()

# Plot Rgs:
fig, ax = plt.subplots()
RG_list = []
for i,n in enumerate(n_arr, start = 0):
    RG_list.append(MeanCosDict[n]['rg'])
    
ax.plot(n_arr, RG_list)

ax.set_xlabel('Degree of Coarse-Graining')
ax.set_ylabel(r'Coarse-Grained Bead Rg $(\sigma)$')


# Plot Cosines
f, axs = plt.subplots(
        nrows=len(n_arr), 
        ncols=1,
        #sharex=True, 
        #sharey=True,
        figsize=(6,4*len(n_arr)),
    )

for i,n in enumerate(n_arr, start = 0):

    ax = axs[i]
    ax.set_title('Bead Size: '+ str(n))
    #ax.plot(MeanCosDict[n]['len'], MeanCosDict[n]['cos'])#, yerr = MeanCosDict[n]['stdcos'])
    ax.errorbar(x=MeanCosDict[n]['len'], y=MeanCosDict[n]['cos'], yerr = MeanCosDict[n]['stdcos']/np.sqrt(numReps))
    ax.set_xlabel(r"Coarse-Grained Contour Length ($\sigma$)")
    ax.set_ylabel("Mean Bond Cosine")
f.tight_layout()    
    
fig = plt.gcf()
fig.savefig('Plots/AllCosines_'+jobName+'.png', dpi = 300)
        

# # Consolidate Bond Length Data
AllBondLens = []

for i in range(1, numReps+1):
    curfile = str(i)+'/output_files/'+bondLenFName
    
    curBondLens = np.genfromtxt(curfile,dtype='f8', delimiter=' ')
    AllBondLens.append(curBondLens)   

AllBondLens = np.matrix(AllBondLens)
MeanBondLens = np.array(AllBondLens.mean(0))[0]
StdBondLens = np.array(AllBondLens.std(0))[0]
np.savetxt('PlotData/'+jobName+'-MeanBondLens-AllReps.dat',[MeanBondLens, StdBondLens],fmt="%s")

fig, ax = plt.subplots()
ax.plot(MeanBondLens);
ax.set_xlabel('Degree of Coarse Graining')
ax.set_ylabel('CG Bond Length')

fig = plt.gcf()
fig.savefig('Plots/BondLength_'+jobName+'.png', dpi = 300)


# # Consolidate Radial Distribution Data
AllHists = []

for i in range(1, numReps+1):
    curfile = str(i)+'/output_files/'+radHistFName
    
    file1 = open(curfile, 'r') 
    Lines = file1.readlines() 
    Edges = [float(edge) for edge in Lines[0].strip().split('    ')]
    AllHists.append([float(val) for val in Lines[1].strip().split('    ')])

AllHists = np.matrix(AllHists)
Edges = np.array(Edges).tolist()
#print(AllHists)
MeanHist = np.array(AllHists.mean(0))[0].tolist()

radHistData = [Edges, MeanHist]
# Store histogram data: 
#first line is bin edges 
#Second line is probability density.
#Leaving Code in general form for later when there will be a different histogram for each force or distance.
f = open('PlotData/'+jobName+'-RadHist-AllReps.dat',"w")

for row in range(len(radHistData)-1):
    for n in radHistData[row]:
        f.write(str(n) + '    ')
    f.write('\n')

for n in radHistData[len(radHistData)-1]:
    f.write(str(n)+ '    ')

f.close()


fig, ax = plt.subplots()

Edges = np.array(Edges)
centers = 0.5*(Edges[:-1]+Edges[1:])
ax.bar(centers, MeanHist, width=.01)
ax.set_xlabel
fig = plt.gcf()
fig.savefig('Plots/RadHist_'+jobName+'.png', dpi = 300)