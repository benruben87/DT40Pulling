#!/usr/bin/env python
# coding: utf-8

# # DT40 15m Native Simulation
# ## Calculates contact map and average PE components for single run.
# ## Also analyzes native length and native dynamics of pull coordinate.

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

platform = 'cuda'
blockSize = 10 #Pull Coordinate is recorded every block
numBlocks = 5*10**5
blocksPerFrame = 100 #Positions of all beads are recorded every Frame.
eqFrames = 10 #Number of frames to skip when calculating mean energy components, contact maps, and bond cosines.
n_arr = [1, 10, 50, 100, 150, 200, 300]
max_tau = 20000

# ## Functions to create pin, slide, and pull forces
# ### Pin and slide are used in this notebook

# In[3]:


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
    pullequation = "-1*f*distance(g1,g2)"

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

#Biases COM of group to the right of xright
def right_force(group, xright =0, frf=100):
    import simtk.openmm as openmm

    pullequation = "frf * step(xright-x1)*(xright-x1)" #Constant force that pushes com to the right of xright
    pullforce = openmm.CustomCentroidBondForce(1, pullequation)
    pullforce.addGlobalParameter('frf', frf)
    pullforce.addGlobalParameter('xright', xright)
    pullforce.addGroup(group)
    pullforce.addBond([0])
    return(pullforce)


# ## Native Simulation of Equilibrated Structure
# Equilibrated structure is loaded from pdb file already placed in directory.

# In[4]:


##Start openMiChroM lib
sim = openmichrom(name='sim', temperature=120)
sim.setup(platform=platform, integrator="Langevin")


# In[5]:


#Folder to save outputs
sim.saveFolder('output_files')


# In[6]:


#read/creation of the initial stat
mypol = sim.create_springSpiral(type_list='input/DT40_chr7.eigen')
#mypol = sim.loadPDB('output_files/input.pdb')


# In[7]:


#Load initial conformation into the system
sim.load(mypol, center=True)
#inputfname = 'Ortd_15m_'+str(rep)+'.pdb'
inputfname = 'ortd.pdb'
sim.setData(sim.load_Coords_PDB(inputfname))#Sets coordinates to oriented structure.
sim_aux = trainertool(TypeList=sim.type_list, state=sim.getData())


# In[8]:


#start HP forces
sim.addGrosbergPolymerBonds(k=30) 
sim.addGrosbergStiffness(k=1.0)
sim.addRepulsiveSoftCore(Et=4.0)


# In[9]:


#MiChroM potentials
#types lambdas = [AA,AB,BB]
type_lambs = sim_aux.getlambfromfile('input/types-DT40-15m')
sim.addTypes_forTrainner(mi=1.51, rc = 2.12, lambdas=type_lambs)


# In[10]:


#Ideal Chromosome potential
#REMEMBER lambs counts all iteraction, the file must have the number of line equals the size of your polymer
#the 3 first lambdas must be zero if want start interaction after 3  neighbor 
sim.lambs = sim_aux.getlambfromfile('input/lambdas_ic_15m')
sim.IC_forTrainner(mi=1.51, rc = 2.12, dcutl=3, dcutupper=735)


# In[11]:


#NO spherical Confinement
#######sim.addSphericalConfinement(density=0.1, k=0.2)


# In[12]:


#if want save initial configuration in pdb format
sim.save(mode = 'pdb', filename = 'loaded_structure.pdp', pdbGroups=sim.chains)


# ### Define pull groups.

# In[13]:


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

r0 = np.sqrt( (first_centroid[0] - second_centroid[0])**2 + 
              (first_centroid[1] - second_centroid[1])**2 +
              (first_centroid[2] - second_centroid[2])**2 )

print("Initial distance between groups is r = {}".format(r0))


# In[14]:


#Pin and slide forces maintain orientation

pin_force_sim = pin_force(g1, kpin=100) #here we using x_pos = 0 and kp = 10000
sim.forceDict["pin"] = pin_force_sim

# Pin right end to positive x-axis
slide_force_sim = slide_force(g2, kslide=100) #here we using x_pos = 0 and kp = 10000
sim.forceDict["slide"] = slide_force_sim


# In[15]:


# group1 = pforce.getGroupParameters(0)[0]
# group2 = pforce.getGroupParameters(1)[0]
# print(group2)


# In[16]:


# pforce.getEnergyFunction()
# pforce.getGlobalParameterDefaultValue(1)


# In[17]:


#define name to save .cndb
sim.initStorage('traj', mode='w')

#Matrix to save energy components:
enerComps = []


# ## No Initial Energy Minimization For This Notebook

# In[18]:


#sim.localEnergyMinimization(tolerance=0.3, maxIterations=0, random_offset=0.02)
#sim.save(mode = 'pdb', pdbGroups=sim.chains, filename = 'EnergyMinimized.pdb')


# ## Function that retrieves pull coordinate:

# In[19]:


def getPCoord(group1, group2, positions):
    first_centroid = np.mean(positions[group1], axis=0) 
    second_centroid = np.mean(positions[group2], axis=0)

    ## calculate r0 distance between groups

    return np.sqrt( (first_centroid[0] - second_centroid[0])**2 )#+ 
                    #(first_centroid[1] - second_centroid[1])**2 +
                    #(first_centroid[2] - second_centroid[2])**2 )


# ## Regular Simulation

# In[20]:


#Attempted to create custom reporter and failed.  Will try just using smaller simulation blocks instead.
#COMForce = CentroidForceReporter( sim.forceDict['pulling'], 50 )
#sim.reporters.append( COMForce )


# In[21]:


#number of blocks simulation
nb=numBlocks
bs = blockSize


# In[22]:


#Matrix to save time
time_record = []

#Matrix to save pull coordinate
pcoord = []

#Matrix to save frame labels
flabels = []

#Append Initial pcoord and time:
time_record.append(0)
pcoord.append(getPCoord(g1, g2, sim.getData()))


# In[23]:


#run simulation
time1 = time.time()

#Save Initial Configuration
sim.save()
flabels.append(sim.step)

#Start Simulation
for t in range(1, nb+1):
    
    sim.doBlock(bs, increment=True, output=(1-np.heaviside(t%blocksPerFrame,0))) #relaxamento
    
    #Record Pulling Data Every Block!
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
     #sim.save(mode = 'pdb', pdbGroups=sim.chains)
time2 = time.time()
print('This run took {:.2f} seconds'.format(time2-time1))


# In[24]:


#close storage file
sim.storage[0].close()

#save last conformation in pdb
sim.save(mode = 'pdb', pdbGroups=sim.chains, filename = 'lastframe.pdb')

# In[25]:


#Save Time and Pull Coordinate to output file.
coordinfo = np.vstack([time_record, pcoord]).T
np.savetxt('output_files/Pull_Coord.txt', coordinfo)

# ## Energy Components
enerComps = np.matrix(enerComps[eqFrames:])
meanEnerComps = np.array(enerComps.mean(0))[0]
enerOutput = [enerNames, meanEnerComps]
print(np.array(meanEnerComps)[0])
with open("output_files/MeanEner.dat", "w") as txt_file:
    for curname in enerNames:
        txt_file.write(curname + "    ")
    txt_file.write("\n")
    
    for curEner in meanEnerComps:
        txt_file.write(str(curEner)+"    ")


