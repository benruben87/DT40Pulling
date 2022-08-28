#!/usr/bin/env python
# coding: utf-8

# # DT40 0m Constant Force Simulation (And Release).
# Loads structure from "initial structures" file and simulates with no added force (and no spherical confinement).
# Saves last frame of simulation to be used for later pulling experiments.
# Saves recording of pull coordinate over time.

# In[1]:


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
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16, 'figure.figsize': [6.0, 5.0]})


# ## Simulation Parameters (To Be Changed Before Running on NOTS):

# In[2]:


#force = FORCEVAL # VARIABLE: FORCEVAL  reduced units.
forcenum = FORCENUM
forcejump = FJUMP
force = forcenum*forcejump

#platform = 'cuda'
platform = 'CPU'
blockSize = 100 #Pull Coordinate is recorded every block, whicih is every 10 steps
numBlocks = 6*10**4
blocksPerFrame = 10 #Positions of all beads are recorded every frame, which is every 50 blocks (500 steps) in this case.

#platform_release = 'cuda'
platform_release = 'CPU'
blockSize_release = 100
numBlocks_release = 2*10**4
blocksPerFrame_release = 10

#Number of frames to skip when calculating mean energy components, contact maps, and bond cosines.
#Should be long enough to allow chromosome to completely relax after application of constant force.
eqFrames = 1500
n_arr = [10, 20, 50, 100, 150, 200, 300, 400]
max_tau = 20000


# ## Function that retrieves pull coordinate:

# In[3]:


def getPCoord(group1, group2, positions):
    first_centroid = np.mean(positions[group1], axis=0) 
    second_centroid = np.mean(positions[group2], axis=0)

    ## calculate r0 distance between groups

    return np.sqrt( (first_centroid[0] - second_centroid[0])**2 )


# ## Functions to create pin, slide, and pull forces
# Pin and slide forces are not working
# constant_pull_force will be used in this notebook.

# In[4]:


def harmonic_pull_force(group1, group2, r0, kp):
    import simtk.openmm as openmm
    
    #r0 = r0 * units.meter * 1e-9
    pullequation = "0.5 * kpull * ((x2-x1)- rp)^2" #Enforces x2>x1

    pullforce = openmm.CustomCentroidBondForce(2, pullequation)

    pullforce.addGlobalParameter('kpull', kp)
    pullforce.addGlobalParameter('rp', r0)
    pullforce.addGroup(group1)
    pullforce.addGroup(group2)
    pullforce.addBond([0,1])
    #pullforce.setForceGroup(8)
    return(pullforce)

def constant_pull_force(group1, group2, f):
    import simtk.openmm as openmm
    
    #Pulls x2 to the right and x1 to the left
    pullequation = "-1*f*(x2-x1)"

    pullforce = openmm.CustomCentroidBondForce(2, pullequation)

    pullforce.addGlobalParameter('f', f)
    pullforce.addGroup(group1)
    pullforce.addGroup(group2)
    pullforce.addBond([0,1])
    return(pullforce)

def pin_force(group1, kpin=100):
    import simtk.openmm as openmm
    
    #r0 = r0 * units.meter * 1e-9
    pullequation = "0.5 * kpin * (x1^2+y1^2+z1^2)"

    pullforce = openmm.CustomCentroidBondForce(1, pullequation)

    print(str(kpin))
    pullforce.addGlobalParameter('kpin', kpin)
    pullforce.addGroup(group1)
    pullforce.addBond([0])
    return(pullforce)

def slide_force(group1, x_min=0, kslide = 100):
    import simtk.openmm as openmm
    
    pullequation = "0.5 * kslide * ((step(x_min-x1)*((x_min-x1)^2))+y1^2+z1^2)"
    #pullequation = "0.5 * kslide * (y1^2+z1^2)"
    pullforce = openmm.CustomCentroidBondForce(1, pullequation)

    pullforce.addGlobalParameter('kslide', kslide)
    pullforce.addGlobalParameter('x_min', x_min)
    pullforce.addGroup(group1)
    pullforce.addBond([0])
    return(pullforce)


# ## Loads Native Output structure and simulates with constant force

# In[5]:


##Start openMiChroM lib
sim = openmichrom(name='sim', temperature=120)
sim.setup(platform=platform, integrator="Langevin")


# In[6]:


#Folder to save outputs
sim.saveFolder('output_files')


# In[7]:


#read/creation of the initial state
mypol = sim.create_springSpiral(type_list='input/DT40_chr7.eigen')


# In[8]:


#Load initial conformation into the system.  Uses output from native sim.
sim.load(mypol, center=True)
sim.setData(sim.load_Coords_PDB('output_files/NativeOutput.pdb'))#Sets coordinates to initial structure.
sim_aux = trainertool(TypeList=sim.type_list, state=sim.getData())


# In[9]:


#start HP forces
sim.addGrosbergPolymerBonds(k=30) 
sim.addGrosbergStiffness(k=1.0)
sim.addRepulsiveSoftCore(Et=4.0)


# In[10]:


#MiChroM potentials
#types lambdas = [AA,AB,BB]
type_lambs = sim_aux.getlambfromfile('input/types-DT40-0m')
sim.addTypes_forTrainner(mi=1.51, rc = 2.12, lambdas=type_lambs)


# In[11]:


#Ideal Chromosome potential
#REMEMBER lambs counts all iteraction, the file must have the number of line equals the size of your polymer
#the 3 first lambdas must be zero if want start interaction after 3  neighbor 
sim.lambs = sim_aux.getlambfromfile('input/lambdas_ic_0m')
sim.IC_forTrainner(mi=1.51, rc = 2.12, dcutl=3, dcutupper=735)


# In[12]:


#NO spherical Confinement
#######sim.addSphericalConfinement(density=0.1, k=0.2)


# In[13]:


#if want save initial configuration in pdb format
sim.save(mode = 'pdb', filename = 'loaded_structure.pdb', pdbGroups=sim.chains)


# ## Add Constant Pull Force

# ### Define pull groups.

# In[14]:


index = [ index for index in range(len(sim.getData())) ]
g1 = index[:50]
g2 = index[-50:]
print(g1,g2)

positions = sim.getData() #get the position of each bead

first_centroid = np.mean(positions[g1], axis=0) 
print(first_centroid)
second_centroid = np.mean(positions[g2], axis=0)
print(second_centroid)

## calculate r0 distance between groups

r0 = np.sqrt( (first_centroid[0] - second_centroid[0])**2)

print("Initial distance between groups is r = {}".format(r0))


# In[15]:


pforce_sim = constant_pull_force(g1,g2, force)
sim.forceDict["pulling"] = pforce_sim

pin_force_sim = pin_force(g1, kpin=100) #here we using x_pos = 0 and kp = 100
sim.forceDict["pin"] = pin_force_sim

# Pin right end to positive x-axis
slide_force_sim = slide_force(g2, kslide=100) #here we using x_pos = 0 and kp = 100
sim.forceDict["slide"] = slide_force_sim


# In[16]:


#define name to save .cndb
sim.initStorage('CF_'+str(forcenum)+'_traj', mode='w')

#Matrix to save energy components:
enerComps = []


# ## CF Simulation

# In[17]:


#Attempted to create custom reporter and failed.  Will try just using smaller simulation blocks instead.
#COMForce = CentroidForceReporter( sim.forceDict['pulling'], 50 )
#sim.reporters.append( COMForce )


# In[18]:


#number of blocks simulation
nb=numBlocks
bs = blockSize


# In[19]:


#Matrix to save time
time_record = []

#Matrix to save pull coordinate
pcoord = []

#Matrix to save frame labels
flabels = []

#Append Initial pcoord and time:
time_record.append(0)
pcoord.append(getPCoord(g1, g2, sim.getData()))


# In[20]:


#run simulation
time1 = time.time()

#Save Initial Configuration
sim.save()
flabels.append(sim.step)

#Start Simulation
for t in range(1, nb+1):
    
    sim.doBlock(bs, increment=True, output=(1-np.heaviside(t%blocksPerFrame,0))) #relaxamento
    
    #Recurd Pulling Data Every Block!
    time_record.append(sim.timestep*t*bs)
    pcoord.append(getPCoord(g1, g2, sim.getData()))
    
    if t% blocksPerFrame == 0:
        ##save trajectory
        sim.save()
        flabels.append(sim.step)
        #if want print forces
        enerNames, curEnerComps = sim.getForces()
        enerComps.append(curEnerComps)
    if t % 200 == 0:
        sim.printForces()
        print("Radius of Gyration: {:.2f}\tBlock: {}/{}".format(sim.RG(), t,nb))
        #Don't save pdb structures throughout simulation in this notebook. 
        #sim.save(mode = 'pdb', pdbGroups=sim.chains)
time2 = time.time()
print('This run took {:.2f} seconds'.format(time2-time1))


# In[21]:


#close storage file
sim.storage[0].close()

#save last conformation in pdb
sim.save(mode = 'pdb', pdbGroups=sim.chains, filename = 'CF_'+str(forcenum)+'_Output.pdb')


# In[22]:


#Save Time and Pull Coordinate to output file.
coordinfo = np.vstack([time_record, pcoord]).T
np.savetxt('output_files/CF_'+str(forcenum)+'_Pull_Coord.txt', coordinfo)


# # Analysis of cf simulation

# ## Mean Energy Components

# In[23]:


enerComps = np.matrix(enerComps[eqFrames:])
meanEnerComps = np.array(enerComps.mean(0))[0]
enerOutput = [enerNames, meanEnerComps]
with open('output_files/MeanEner_'+str(forcenum)+'.dat', "w") as txt_file:
    for curname in enerNames:
        txt_file.write(curname + "    ")
    txt_file.write("\n")
    
    for curEner in meanEnerComps:
        txt_file.write(str(curEner)+"    ")


# # Release Simulation:

# In[24]:


##Start openMiChroM lib
release = openmichrom(name='release', temperature=120)
release.setup(platform=platform_release, integrator="Langevin")


# In[25]:


#Folder to save outputs
release.saveFolder('output_files')


# In[26]:


#read/creation of the initial state
mypol = release.create_springSpiral(type_list='input/DT40_chr7.eigen')


# In[27]:


#Load initial conformation into the system.  Uses output from native sim.
release.load(mypol, center=True)
release.setData(sim.load_Coords_PDB('output_files/CF_'+str(forcenum)+'_Output.pdb'))#Sets coordinates to initial structure.
release_aux = trainertool(TypeList=release.type_list, state=release.getData())


# In[28]:


#start HP forces
release.addGrosbergPolymerBonds(k=30) 
release.addGrosbergStiffness(k=1.0)
release.addRepulsiveSoftCore(Et=4.0)


# In[29]:


#MiChroM potentials
#types lambdas = [AA,AB,BB]
type_lambs = release_aux.getlambfromfile('input/types-DT40-0m')
release.addTypes_forTrainner(mi=1.51, rc = 2.12, lambdas=type_lambs)


# In[30]:


#Ideal Chromosome potential
#REMEMBER lambs counts all iteraction, the file must have the number of line equals the size of your polymer
#the 3 first lambdas must be zero if want start interaction after 3  neighbor 
release.lambs = sim_aux.getlambfromfile('input/lambdas_ic_0m')
release.IC_forTrainner(mi=1.51, rc = 2.12, dcutl=3, dcutupper=735)


# In[31]:


#NO spherical Confinement
#######sim.addSphericalConfinement(density=0.1, k=0.2)


# In[32]:


#if want save initial configuration in pdb format
#release.save(mode = 'pdb', filename = 'loaded_structure.pdb', pdbGroups=sim.chains)


# In[33]:


#Checks Initial Pull Coordinate.
positions = release.getData() #get the position of each bead

first_centroid = np.mean(positions[g1], axis=0) 
print(first_centroid)
second_centroid = np.mean(positions[g2], axis=0)
print(second_centroid)

## calculate r0 distance between groups

r0 = np.sqrt( (first_centroid[0] - second_centroid[0])**2 )

print("Length Before Release is r = {}".format(r0))


# In[34]:


#Add Pin and Slide Forces
pin_force_rel = pin_force(g1, kpin=100) #here we using x_pos = 0 and kp = 10000
release.forceDict["pin"] = pin_force_rel

# Pin right end to positive x-axis
slide_force_rel = slide_force(g2, kslide=100) #here we using x_pos = 0 and kp = 10000
release.forceDict["slide"] = slide_force_rel


# In[35]:


#define name to save .cndb
release.initStorage('Release_'+str(forcenum)+'_traj', mode='w')

#Matrix to save energy components:
enerComps = []


# In[36]:


#number of blocks simulation
nb=numBlocks_release
bs = blockSize_release
blocksPerFrame = blocksPerFrame_release


# In[37]:


#Matrix to save time
time_record_release = []

#Matrix to save pull coordinate
pcoord_release = []

#Matrix to save frame labels
flabels_release = []

#Append Initial pcoord and time:
time_record_release.append(0)
pcoord_release.append(getPCoord(g1, g2, release.getData()))


# In[38]:


#run simulation
time1 = time.time()

#Save Initial Configuration
release.save()
flabels_release.append(sim.step)

#Start Simulation
for t in range(1, nb+1):
    
    release.doBlock(bs, increment=True, output=(1-np.heaviside(t%blocksPerFrame,0))) #relaxamento
    
    #Recurd Pulling Data Every Block!
    time_record_release.append(sim.timestep*t*bs)
    pcoord_release.append(getPCoord(g1, g2, release.getData()))
    
    if t% blocksPerFrame == 0:
        ##save trajectory
        release.save()
        flabels_release.append(release.step)
        #if want print forces
        enerNames, curEnerComps = release.getForces()
        enerComps.append(curEnerComps)
    if t % 200 == 0:
        release.printForces()
        print("Radius of Gyration: {:.2f}\tBlock: {}/{}".format(release.RG(), t,nb))
        #Don't save pdb structures throughout simulation in this notebook. 
        #sim.save(mode = 'pdb', pdbGroups=sim.chains)
time2 = time.time()
print('This run took {:.2f} seconds'.format(time2-time1))


# In[39]:


#close storage file
release.storage[0].close()

#save last conformation in pdb
release.save(mode = 'pdb', pdbGroups=sim.chains, filename = 'Release_'+str(forcenum)+'_Output.pdb')


# In[40]:


#Save Time and Pull Coordinate to output file.
coordinfo_release = np.vstack([time_record_release, pcoord_release]).T
np.savetxt('output_files/Release_'+str(forcenum)+'_Pull_Coord.txt', coordinfo_release)


# # Plot Pull Coordinate (Testing Purposes Only)

# In[41]:


#all_times = release.timestep*np.array(range(len(pcoord)+len(pcoord_release)-1))
#all_pcoord = pcoord + pcoord_release[1:]

#fig, ax = plt.subplots()
#ax.plot(all_times, all_pcoord)
#ax.set_xlabel(r"Time ($\tau$)")
#ax.set_ylabel(r"End to End Distance ($\sigma$)")
#ax.set_title('Force = '+ str(forcenum))
#ax.axvline(x=all_times[len(pcoord)-1], ls='--', color = 'r')