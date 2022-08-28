#!/usr/bin/env python
# coding: utf-8

# # Constant Force Simulation Analysis Scripts
# Combines data outputs from runs in folders 1,2,.... numReps to plot means over multiple replicas, comparing all force values.  
# Contact maps, bond cosines, and energy components
# Saves all plotted data to filesto be used for making plots combining various chromosomes later.

# In[1]:


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
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#Don't show figures, only save:
mpl.use('Agg')

plt.rcParams.update({'font.size': 16, 'figure.figsize': [6.0, 5.0]})
#plt.rcParams.update({'font.size': 16})


if not os.path.exists('PlotData'):
    os.makedirs('PlotData')
    
if not os.path.exists('Plots'):
    os.makedirs('Plots')


# In[28]:


# Variables:
numBeads = 738
numReps = 40 
jobName = 'CF15m'
Time_Label = '15m'
eqBlocks = 15000 #Number of simulation blocks to skip before recording pull coordinate for histograms.

forcenum = 5 
forcejump = 1
timestep=.01
blockSize = 100

dt = timestep*blockSize
forces = np.arange(0,forcejump*forcenum+forcejump, forcejump)

NoForcePullCoordFName = 'Native_Pull_Coord.txt'

def makeBondLenFName(forcenum):
    return 'BondLens_' + str(forcenum)+ '.txt'

def makeRadHistFName(forcenum):
    return 'RadHist_' + str(forcenum) + '.txt'

def makeEnerFName(forcenum):
    return 'MeanEner_' + str(forcenum) + '.dat'

def makeContMapFName(forcenum):
    return 'Prob_' + str(forcenum) + '.dat'

def makeCosFName(forcenum):
    return 'CosDict_'+str(forcenum)+'.txt'

def makeMSDFName(forcenum):
    return 'CF_'+str(forcenum)+'_MSD.txt'
    
def makePullCoordFName(forcenum):
    return 'CF_'+str(forcenum)+'_Pull_Coord.txt'

def makeReleasePullCoordFName(forcenum):
    return 'Release_'+str(forcenum)+'_Pull_Coord.txt'

def makeSingleJobName(forcenum):
    return 'CF15m_'+str(forcenum)


# In[3]:


#List of Colors:
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab: gray', 'tab:olive', 'tab:cyan']


# # Contact Map and P vs D:

# In[4]:


#Contact Maps:
AllProbs = []
for curfnum in range(forcenum+1):
    
    contMapFName = makeContMapFName(curfnum)
    singlejobname = makeSingleJobName(curfnum)
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

    np.savetxt('PlotData/'+singlejobname+'-Dist-all.dat',r,fmt="%s")
    fig, ax = plt.subplots()
    ax.matshow(r,norm=mpl.colors.LogNorm(vmin=0.0001, vmax=r.max()),cmap="Reds")
    #fig.show()

    fig = plt.gcf()
    fig.savefig('Plots/ContMap_'+singlejobname+'.png', dpi = 300)

    Prob = []
    for i in range(l):
        ps = [r[j, j+i] for j in range(l-i)]
        Prob.append(np.mean(np.array(ps)))
    
    AllProbs.append(np.array(Prob))
    sys.stdout.flush()
    end = time.time()
    elapsed = end - start
    print("Ran in %f sec" % elapsed)
    

# Save and plot all P vs D curves:    
np.savetxt('PlotData/' + jobName+'-PvD-all.dat',np.matrix([np.array(range(l))]+ AllProbs),fmt="%s")
    
fig, ax = plt.subplots()

for curfnum in range(forcenum+1):
    curforce = forcejump*curfnum
    ax.plot(50*np.array(range(l)),AllProbs[curfnum], label=str(curforce), c = colors[curfnum])
    
ax.set_xlabel(r"Genomic Distance (kb)")
ax.set_ylabel('Mean Contact Probability')
ax.set_yscale('log')
ax.legend(title=r"Force ($\epsilon/\sigma$)")
fig.tight_layout()
fig = plt.gcf()
fig.savefig('Plots/AllPvD_'+jobName+'.png')


# # Energy Components:

# In[5]:


MeanEnergies_AllForces = []
StdEnergies_AllForces = []

for curfnum in range(forcenum+1):
    
    enerFName = makeEnerFName(curfnum)
    singlejobname = makeSingleJobName(curfnum)

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
    np.savetxt('PlotData/'+singlejobname+'-MeanEnergies-AllReps.dat',[EnergyNames, MeanEnergies, StdEnergies],fmt="%s")

    MeanEnergies_AllForces.append(MeanEnergies)
    StdEnergies_AllForces.append(StdEnergies)

    
MeanEnergies_AllForces = np.matrix(MeanEnergies_AllForces)
StdEnergies_AllForces = np.matrix(StdEnergies_AllForces)
fig, ax = plt.subplots(figsize = (10,4))

EnergyComponents = ['Bonds', 'Angles', 'Soft Core', 'Type-Type', 'Ideal Chromosome']

for index, curcomp in enumerate(EnergyComponents):
    curenergies = (np.array((MeanEnergies_AllForces[:,index]-MeanEnergies_AllForces[0,index]).T)[0])
    curstds = ((np.array(StdEnergies_AllForces[:,index])).T)[0]
    ax.errorbar(forces, curenergies, yerr = curstds, label = curcomp, capsize=2)
    
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel(r"Energy ($\epsilon$)")
ax.set_xlabel(r"Force ($\epsilon/\sigma$)")
ax.set_title(jobName)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

#fig.show()
fig.tight_layout()
fig = plt.gcf()
fig.savefig('Plots/Energies_'+jobName+'.png', dpi = 300)


# # Bond Cosines Under Force

# In[7]:


# importing the module 
import ast 
AllMeanCosDicts = [] #List of Mean Cosine Dictionaries.

for curfnum in range(forcenum+1):
    
    cosFName = makeCosFName(curfnum)
    singlejobname = makeSingleJobName(curfnum)
    AllCosDicts = []

    for i in range(1, numReps+1):
        curfile = str(i)+'/output_files/'+cosFName
        # reading the data from the file 
        with open(curfile) as f: 
            data = f.read() 
        print("Data type before reconstruction: ", type(data)) 

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
    f = open('PlotData/'+singlejobname+'MeanCosDict.txt',"w")
    f.write( str(MeanCosDict) )
    f.close()
    AllMeanCosDicts.append(MeanCosDict)


# In[8]:


f, axs = plt.subplots(
        nrows=len(n_arr), 
        ncols=1,
        #sharex=True, 
        #sharey=True,
        figsize=(8,4*len(n_arr)),
    )

for i,n in enumerate(n_arr, start = 0):

    ax = axs[i]
    ax.set_title('Bead Size: '+ str(n))
    
    for index, force in enumerate(forces):
        MeanCosDict = AllMeanCosDicts[index]
        #ax.errorbar(x=MeanCosDict[n]['len'], y=MeanCosDict[n]['cos'], yerr = MeanCosDict[n]['stdcos'], label = force, c = colors[index])
        ax.plot(MeanCosDict[n]['len'], MeanCosDict[n]['cos'], label = force, c = colors[index])
        
    ax.set_xlabel(r"Coarse-Grained Contour Length ($\sigma$)")
    ax.set_ylabel("Mean Bond Cosine")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=r'Force $(\epsilon/\sigma)$')
f.tight_layout()    
    
fig = plt.gcf()
fig.savefig('Plots/AllCosines_'+jobName+'.png', dpi = 300)    


# In[13]:


# Plot Rgs:
fig, ax = plt.subplots()    
    
for i,n in enumerate(n_arr, start = 0):
    RG_list = []
    for index, force in enumerate(forces):
        RG_list.append(AllMeanCosDicts[index][n]['rg'])
    
    ax.plot(forces, RG_list, label = str(n))

ax.set_xlabel(r'Force $(\epsilon/\sigma)$')
ax.set_ylabel(r'Coarse-Grained Bead Rg $(\sigma)$')
ax.legend(title = 'Beads Per Group', bbox_to_anchor=(1.05, 1), loc='upper left')
fig = plt.gcf()
fig.savefig('Plots/RGs_'+jobName+'.png', dpi = 300)


# ## Combine MSD's into one plot.

# In[14]:


Taus = []
AllMeanMSDs = []

for curfnum in range(forcenum+1):
    MSDFName = makeMSDFName(curfnum)
    MeanMSD = []
    for i in range(1, numReps+1):
        curfile = str(i)+'/output_files/'+MSDFName
        MSD_Data = np.genfromtxt(curfile,dtype='f8', delimiter=' ')
        Taus = MSD_Data[:,0]
        MeanMSD.append(MSD_Data[:,1])

    MeanMSD = np.matrix(MeanMSD)
    MeanMSD = np.array(MeanMSD.mean(0))[0]
    AllMeanMSDs.append(MeanMSD.T)
    MeanMSD_Data = np.vstack([Taus.T] + AllMeanMSDs)

    
np.savetxt('PlotData/'+jobName+'-MSD.dat',MeanMSD_Data,fmt="%s")
fig, ax = plt.subplots(figsize = (7,4))

for curfnum in range(forcenum+1):
    ax.plot(Taus, AllMeanMSDs[curfnum], label = forces[curfnum], c = colors[curfnum])
    
ax.set_xlabel(r"Time Difference ($\tau$)")
ax.set_ylabel(r"MSD ($\sigma^2$)")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=r'Force $(\epsilon/\sigma)$')
fig.tight_layout()
fig = plt.gcf()
fig.savefig('Plots/MSD_'+jobName+'.png')


# ## Make Histogram of pull coordinate and save bin edges and counts.

# In[20]:


BinWidth = .1
Edges = np.arange(0, 40+BinWidth, BinWidth)

histData = [Edges]

f, axs = plt.subplots(
        nrows=len(forces), 
        ncols=1,
        sharex=True, 
        sharey=True,
        figsize=(6,1*len(n_arr)),
    )

for curfnum in range(forcenum+1):
    
    force = forcejump*curfnum
    PullCoordFName = makePullCoordFName(curfnum)
    
    AllCoords = []

    for i in range(1, numReps+1):
        curfile = str(i)+'/output_files/'+PullCoordFName
        Coord_Data = np.genfromtxt(curfile,dtype='f8', delimiter=' ')
        AllCoords.append(Coord_Data[eqBlocks:,1])

    AllCoords=np.hstack(AllCoords)
    ax = axs[curfnum]
    binValues,binEdges,c = ax.hist(AllCoords, bins=Edges, density=True, color=colors[curfnum])
    histData.append(binValues.tolist())

f.text(0.5, 0, r"End to End Distance ($\sigma$)", ha='center')
f.text(-.05, 0.5, r"Probability Density", va='center', rotation='vertical')

#f.tight_layout()    
fig = plt.gcf()
fig.savefig('Plots/Hists_'+jobName+'.png')    

# Store histogram data: 
#first line is bin edges 
#Following lines are probability densities for each force.
f = open('PlotData/'+jobName+"HistData.txt","w")

for row in range(len(histData)-1):
    for n in histData[row]:
        f.write(str(n) + '    ')
    f.write('\n')

for n in histData[len(histData)-1]:
    f.write(str(n)+ '    ')

f.close()


# # Mean Force Extension Curve:
# Plots Mean and Standard Deviation of Pulling Coodinate at each force and saves data to text file.

# In[21]:


MeanExtensions = []
StdExtensions = []
SEMExtensions = []    

for curfnum in range(forcenum+1):
    
    force = forcejump*curfnum
    PullCoordFName = makePullCoordFName(curfnum)
    
    AllCoords = []
    AllMeanCoords = []

    for i in range(1, numReps+1):
        curfile = str(i)+'/output_files/'+PullCoordFName
        Coord_Data = np.genfromtxt(curfile,dtype='f8', delimiter=' ')
        AllCoords.append(Coord_Data[eqBlocks:,1])
        AllMeanCoords.append(np.mean(Coord_Data[eqBlocks:, 1]))

    AllCoords=np.hstack(AllCoords)
    
    MeanExtensions.append(np.mean(AllMeanCoords))
    StdExtensions.append(np.std(AllCoords))
    SEMExtensions.append(np.std(AllMeanCoords)/np.sqrt(len(AllMeanCoords)))

#Linear Regression of Force Extension Curve:
x = np.array(MeanExtensions).reshape((-1,1))
y=np.array(forces)
model = LinearRegression()
model.fit(x,y)
r_sq = model.score(x,y)
intercept = model.intercept_
slope = float(model.coef_)

print('Y-Intercept: ',intercept)
print('Slope: ', slope)
print('r squared: ', r_sq)
NatLen = float(-intercept/slope)
print('Native Length: ', NatLen)

    
fig, ax = plt.subplots()
ax.errorbar(MeanExtensions, forces, xerr = StdExtensions, ls='None', elinewidth=2, capsize=6, capthick=2)
ax.errorbar(MeanExtensions, forces, xerr = SEMExtensions, ls='None', elinewidth=2, capsize=4, fmt = 'ro')
ax.plot(MeanExtensions, np.multiply(slope,MeanExtensions)+intercept, ls = ':', color = 'r')
ax.set_xlabel(r'Extension ($\sigma$)')
ax.set_ylabel(r'Force ($\epsilon/\sigma$)')
ax.set_title('Force-Extension Curve')

NatLenRounded = float(round(10*NatLen))/10
SlopeRounded = float(round(100*slope))/100
#Display Slope and Native Length on Plot:
left, right = ax.get_xlim()  # return the current xlim
bottom, top = ax.get_ylim()

ax.text(left+.1, top-.3, s='X-Int: ' + str(NatLenRounded)+r" $\sigma$")
ax.text(left+.1, top-.7, s='Slope: ' + str(SlopeRounded) + r" $\epsilon/\sigma^2$")
ax.text(left+.1, top-1.1, s=r'$r^2$: ' + str(r_sq))



fig.tight_layout()
fig = plt.gcf()
fig.savefig('Plots/FExt_'+jobName+'.png')    

# Store force extension data: 
#first line is force
#Second Line is mean extension
#Third Line is STD
#Fourth Line is SEM

np.savetxt('PlotData/'+jobName+'-FExtDat.dat',[forces, MeanExtensions, StdExtensions, SEMExtensions],fmt="%s")


# # Tophat Force Extension Time Courses

# In[22]:


#First gets mean Coordinate from NoForce Simulations:
AllCoords_NF = []
for i in range(1, numReps+1):
        curfile = str(i)+'/output_files/'+NoForcePullCoordFName
        Coord_Data_NF = np.genfromtxt(curfile,dtype='f8', delimiter=' ')
        AllCoords_NF.append(Coord_Data_NF[:,1].T)

MeanCoord_NF = np.mean(AllCoords_NF, axis = 0)
time_nf = np.multiply(dt,range(len(MeanCoord_NF)))
np.savetxt('PlotData/NoForce-'+jobName+'-MeanExtTrace.dat',[time_nf, MeanCoord_NF],fmt="%s")
CFTime = dt*(len(MeanCoord_NF)-1)

#Then gets mean coordinate from each constant force simulation
MeanExts_CF = []

for curfnum in range(forcenum+1):
    
    force = forcejump*curfnum
    PullCoordFName = makePullCoordFName(curfnum)
    
    AllCoords_CF = []

    for i in range(1, numReps+1):
        curfile = str(i)+'/output_files/'+PullCoordFName
        Coord_Data_CF = np.genfromtxt(curfile,dtype='f8', delimiter=' ')
        AllCoords_CF.append(Coord_Data_CF[:,1].T)

    CurMeanExt_CF = np.mean(AllCoords_CF, axis=0)    
    MeanExts_CF.append(CurMeanExt_CF)

time_cf = np.multiply(dt,range(len(MeanExts_CF[0])))
np.savetxt('PlotData/CF-'+jobName+'-MeanExtTrace.dat',[time_cf]+MeanExts_CF,fmt="%s")
RelTime = CFTime+dt*(len(MeanExts_CF[0])-1)
#Finally gets mean coordinate from relaxation experiments.

MeanExts_Rel = []

for curfnum in range(forcenum+1):
    
    force = forcejump*curfnum
    PullCoordFName = makeReleasePullCoordFName(curfnum)
    
    AllCoords_Rel = []

    for i in range(1, numReps+1):
        curfile = str(i)+'/output_files/'+PullCoordFName
        Coord_Data_Rel = np.genfromtxt(curfile,dtype='f8', delimiter=' ')
        AllCoords_Rel.append(Coord_Data_Rel[:,1].T) # 

    CurMeanExt_Rel = np.mean(AllCoords_Rel, axis=0)    
    MeanExts_Rel.append(CurMeanExt_Rel)


time_rel = np.multiply(dt,range(len(MeanExts_Rel[0])))
np.savetxt('PlotData/Rel-'+jobName+'-MeanExtTrace.dat',[time_rel]+MeanExts_Rel,fmt="%s")
    
#Concatenates mean trajectories into full tophat experimental timecourses
FullMeanExts = []
for curfnum in range(forcenum+1):
    curFullMeanExt = np.hstack([MeanCoord_NF, MeanExts_CF[curfnum][1:], MeanExts_Rel[curfnum][1:]])
    FullMeanExts.append(curFullMeanExt)
    
time = np.linspace(0, dt*len(FullMeanExts[0]), len(FullMeanExts[0]), endpoint = False)
    
leadingTime = 1000
followingTime =4000 

CFIndices = range(int((CFTime-leadingTime)/dt), int((CFTime+followingTime)/dt)+1)
RelIndices = range(int((RelTime-leadingTime)/dt), int((RelTime+followingTime)/dt)+1)

fig = plt.figure(figsize = (9,8))

ax = fig.add_subplot(211)

for i, force in reversed(list(enumerate(forces))):
    ax.plot(time, FullMeanExts[i], label = force, color = colors[i])

ax.set_xlabel(r'Time ($\tau$)')
ax.set_ylabel(r'Extension ($\sigma$)')
ax.set_title('Stress Relaxation Experiment')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=r'Force $(\epsilon/\sigma)$')

ax.axvline(x=CFTime, linestyle = ':', c='red')
ax.axvline(x=RelTime, linestyle = ':', c='red')

bottom, top = ax.get_ylim()                         
# For visualization purposes we mark the bounding boxes with rectangles
ax.add_patch(plt.Rectangle((CFTime-leadingTime, bottom+0.5), (followingTime+leadingTime), top-bottom-1, ls="--", ec="c", fc="None"))
ax.add_patch(plt.Rectangle((RelTime-leadingTime, bottom+0.5), (followingTime+leadingTime), top-bottom-1, ls="--", ec="m", fc="None"))

#Adds plot around force application.
ax = fig.add_subplot(223)
for i, force in reversed(list(enumerate(forces))):
    ax.plot(time[CFIndices], FullMeanExts[i][CFIndices], label = force, color = colors[i])

ax.set_xlabel(r'Time ($\tau$)')
ax.set_ylabel(r'Extension ($\sigma$)')
ax.add_patch(plt.Rectangle((CFTime-leadingTime, bottom+0.5), (followingTime+leadingTime), top-bottom-1, ls="--", ec="c", fc="None"))
ax.axvline(x=CFTime, linestyle = ':', c='red')

          
#Adds plot around force application.
ax = fig.add_subplot(224)
for i, force in reversed(list(enumerate(forces))):
    ax.plot(time[RelIndices], FullMeanExts[i][RelIndices], label = force, color = colors[i])

ax.set_xlabel(r'Time ($\tau$)')
ax.set_ylabel(r'Extension ($\sigma$)')
ax.add_patch(plt.Rectangle((RelTime-leadingTime, bottom+0.5), (followingTime+leadingTime), top-bottom-1, ls="--", ec="m", fc="None"))
ax.axvline(x=RelTime, linestyle = ':', c='red')

fig.tight_layout()
fig = plt.gcf()
fig.savefig('Plots/Tophat_'+jobName+'.png')    

# Store extension data: 
#first line is time
#Remaining lines are average extensions under pulling forces, in increasing order of pull force

np.savetxt('PlotData/TophatExt'+jobName+'-MeanExtTrace.dat',[time]+ FullMeanExts,fmt="%s")


# # Hi-C Difference Maps

# In[24]:


tiny  = 10**-12

#Contact Maps:
AllProbs = []

NativeFName = 'CF15m_'+str(0)+'-Dist-all.dat'
NativeMap = np.genfromtxt('PlotData/'+NativeFName,dtype='f8', delimiter=' ')

for curfnum in range(forcenum+1):
    filename = 'CF15m_'+str(curfnum)+'-Dist-all.dat'
    r=np.genfromtxt('PlotData/'+filename,dtype='f8', delimiter=' ')

    print(r.max(),r.min())
    sys.stdout.flush()

    LogRatioMap = np.log2(r+tiny)-np.log2(NativeMap+tiny)
    diffMap = r-NativeMap
    fig, ax = plt.subplots()
    #cax = ax.matshow(diffMap,cmap='seismic')
    cax = ax.matshow(diffMap,norm=mpl.colors.Normalize(vmin=-.2, vmax=.2),cmap='seismic')
    fig.colorbar(cax)
    #fig.show()

    fig = plt.gcf()
    fig.savefig('Plots/DiffMap_15m_'+str(curfnum)+'.png', dpi = 300)


# # Radial Histograms

# In[30]:


AllMeanHists = []

for curfnum in range(forcenum+1):
    
    AllHists = []

    for i in range(1, numReps+1):
        curfile = str(i)+'/output_files/'+makeRadHistFName(curfnum)
        file1 = open(curfile, 'r') 
        Lines = file1.readlines() 
        Edges = [float(edge) for edge in Lines[0].strip().split('    ')]
        AllHists.append([float(val) for val in Lines[1].strip().split('    ')])

    AllHists = np.matrix(AllHists)
    Edges = np.array(Edges).tolist()
    #print(AllHists)
    MeanHist = np.array(AllHists.mean(0))[0].tolist()
    AllMeanHists.append(MeanHist)
    
    fig, ax = plt.subplots()
    Edges = np.array(Edges)
    centers = 0.5*(Edges[:-1]+Edges[1:])
    ax.bar(centers, MeanHist, width=.01)
    ax.set_xlabel('Distance From Core')
    fig = plt.gcf()
    fig.savefig('Plots/RadHist_'+str(curfnum)+'_'+jobName+'.png', dpi = 300)

radHistData = [Edges]+AllMeanHists

# Store histogram data: 
#first line is bin edges 
#Remaining lines are probability densities.
#Leaving Code in general form for later when there will be a different histogram for each force or distance.
f = open('PlotData/'+jobName+'-RadHist-AllReps.dat',"w")

for row in range(len(radHistData)-1):
    for n in radHistData[row]:
        f.write(str(n) + '    ')
    f.write('\n')

for n in radHistData[len(radHistData)-1]:
    f.write(str(n)+ '    ')

f.close()


AllMeanBondLens = []

for curfnum in range(forcenum+1):
    
    bondLenFName = makeBondLenFName(curfnum)
    singlejobname = makeSingleJobName(curfnum)
    l=numBeads

    curBondLens = []
    
    for i in range(1, numReps+1):

        curfile = str(i)+'/output_files/'+bondLenFName
        curBondLens.append(np.genfromtxt(curfile,dtype='f8', delimiter=' '))
    
    curBondLens = np.mean(curBondLens, axis = 0)
    AllMeanBondLens.append(curBondLens)

np.savetxt('PlotData/'+jobName+'-AllBondLens.dat',AllMeanBondLens,fmt="%s")
