#!/usr/bin/env python
# coding: utf-8

# # DT40 0m No Force Simulation.
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

#platform = 'cuda'
platform = 'CPU'
blockSize = 100 #Pull Coordinate is recorded every block
numBlocks = 2*10**4
blocksPerFrame = 10 #Positions of all beads are recorded every Frame.
eqFrames = 1 #Number of frames to skip when calculating mean energy components, contact maps, and bond cosines.

Init_Struct_FName = 'ortd.pdb'


# ## Functions to create pin, slide, and pull forces
# Pin and slide forces are not working
# Not used in this notebook, but will be used for CF simulations.

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


# ## Loads Init structure and simulates

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


# In[7]:


#Load initial conformation into the system
sim.load(mypol, center=True)
sim.setData(sim.load_Coords_PDB(Init_Struct_FName))#Sets coordinates to initial structure.
sim_aux = trainertool(TypeList=sim.type_list, state=sim.getData())


# In[8]:


#start HP forces
sim.addGrosbergPolymerBonds(k=30) 
sim.addGrosbergStiffness(k=1.0)
sim.addRepulsiveSoftCore(Et=4.0)


# In[9]:


#MiChroM potentials
#types lambdas = [AA,AB,BB]
type_lambs = sim_aux.getlambfromfile('input/types-DT40-0m')
sim.addTypes_forTrainner(mi=1.51, rc = 2.12, lambdas=type_lambs)


# In[10]:


#Ideal Chromosome potential
#REMEMBER lambs counts all iteraction, the file must have the number of line equals the size of your polymer
#the 3 first lambdas must be zero if want start interaction after 3  neighbor 
sim.lambs = sim_aux.getlambfromfile('input/lambdas_ic_0m')
sim.IC_forTrainner(mi=1.51, rc = 2.12, dcutl=3, dcutupper=735)


# In[11]:


#NO spherical Confinement
#######sim.addSphericalConfinement(density=0.1, k=0.2)


# In[12]:


#if want save initial configuration in pdb format
sim.save(mode = 'pdb', filename = 'loaded_structure.pdb', pdbGroups=sim.chains)


# ### Define pull groups

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

r0 = np.sqrt( (first_centroid[0] - second_centroid[0])**2 )

print("Initial distance between groups is r = {}".format(r0))


# In[14]:


#Add Pin and Slide Forces
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
sim.initStorage('NativeTraj', mode='w')

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

    return np.sqrt( (first_centroid[0] - second_centroid[0])**2 )


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


# In[24]:


#close storage file
sim.storage[0].close()

#save last conformation in pdb
sim.save(mode = 'pdb', pdbGroups=sim.chains, filename = 'NativeOutput.pdb')


# In[25]:


#Save Time and Pull Coordinate to output file.
coordinfo = np.vstack([time_record, pcoord]).T
np.savetxt('output_files/Native_Pull_Coord.txt', coordinfo)


# # Analysis
# 
# ## Contact Maps.  Not calculated here.

# In[26]:


# import numpy as np
# import time
# import h5py
# from   scipy.spatial  import distance


# In[27]:


# start         = time.time()

# mu            = 1.51       # parameter for contact function f
# r_cut         = 2.12       # parameter for contact function f

# chro = [7,7]

# res  = 50000


# In[28]:


# chro1 = h5py.File('output_files/NativeNativeTraj_0.cndb', 'r') # comando para abrir o arquivo
# chro2 = h5py.File('output_files/NativeNativeTraj_0.cndb', 'r') # comando para abrir o arquivo

# frames = len(chro1.keys()) - 1
# firstkey = list(chro1.keys())[0]

# print('1st Chromosome: Chro {:} with {:} beads'.format(chro[0], (len(chro1[firstkey]))))
# print('2nd Chromosome: Chro {:} with {:} beads'.format(chro[1], (len(chro2[firstkey]))))


# In[29]:


# P = np.zeros((len(chro1[firstkey]), len(chro2[firstkey])))

# for i,label in enumerate(flabels):
#     XYZ1 = np.asarray(chro1[str(label)])
#     XYZ2 = np.asarray(chro2[str(label)])
#     if i >eqFrames:
#      D = distance.cdist(XYZ1, XYZ2, 'euclidean')
#      D[D<=1.0] = 0.0
#      P += 0.5 * (1.0 + np.tanh(mu * (r_cut - D)))

#     end     = time.time()
#     elapsed = end - start
#     #print('Frame {:} : %.3f s'.format(i) % elapsed)

#     if i % 500 == 0:
#         print("Reading frame", i)

# np.savetxt('output_files/Prob{:}_{:}.dat'.format(chro[0], chro[1]), np.divide(P , i-eqFrames))

# chro1.close()
# chro2.close()

# end     = time.time()
# elapsed = end - start

# print('Total frames: {:}'.format(len(flabels)))
# print('File saved: Prob{:}_{:}.dat\n'.format(chro[0], chro[1]))
# print('Ran in %.3f sec' % elapsed)
# print('############################################################')


# ## Energy Components (Not Calculated Here)

# In[30]:


# enerComps = np.matrix(enerComps[eqFrames:])
# meanEnerComps = np.array(enerComps.mean(0))[0]
# enerOutput = [enerNames, meanEnerComps]
# print(np.array(meanEnerComps)[0])
# with open("output_files/MeanEner.dat", "w") as txt_file:
#     for curname in enerNames:
#         txt_file.write(curname + "    ")
#     txt_file.write("\n")
    
#     for curEner in meanEnerComps:
#         txt_file.write(str(curEner)+"    ")


# ## Bending Analysis (Coarse-Grained Skeletons) (Not Calculated Here)

# In[31]:


# import numpy as np
# import sys
# from numpy.random import random as rand
# from numpy.linalg import norm
# from itertools import islice, product, combinations
# import matplotlib.pyplot as plt
# import time
# import os


# In[32]:


# def calc_stiffness(pos,n):
#     renrm=[]
#     pos=np.array(pos)
#     #print pos.shape

#     #update renrm to contain the renomralized polymer positions
#     for xx in range(len(pos)//n): 
#         rcm=np.array([np.mean(pos[xx*n:(xx+1)*n,jj]) for jj in range(3)])
#         renrm.append(rcm)
#     renrm=np.array(renrm)
#     #print(renrm.shape)
    
#     #agv_p contains the cosine of angle as a function of contour length (in units of renormalized beads) 
#     #... stored as keys in dict
#     avg_p={}
#     for xx in range(renrm.shape[0]-2):
#         a=renrm[xx+1,:]-renrm[xx,:]
#         #print a
#         for yy in range(xx+1,renrm.shape[0]-1):
#             b=renrm[yy+1,:]-renrm[yy,:]
#             #print abs(xx-yy),np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
#             if abs(xx-yy) in avg_p.keys():
#                 avg_p[abs(xx-yy)].append((np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)))
#             else: 
#                 avg_p[abs(xx-yy)]=[(np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b))]
#     #print avg_p.keys()
    
#     cos_mean,cos_std=[],[]
    
#     #print(np.sort(list(avg_p.keys())))
#     for key in np.sort(list(avg_p.keys())): 
#         cos_mean.append(np.mean(avg_p[key]))
#         cos_std.append(np.std(avg_p[key]))
    
#     #Calculates Mean Distance Between Renormalized Beads:
    
#     meanBondLength = []
#     for xx in range(renrm.shape[0]-2):
#         a=renrm[xx+1,:]-renrm[xx,:]
#         meanBondLength.append(np.linalg.norm(a))
    
#     meanBondLength = np.mean(meanBondLength)
#     ContourLens = meanBondLength*np.sort(list(avg_p.keys()))
#     return ContourLens,cos_mean,cos_std

# def calc_rg(X, axes = np.ones((3,1))):
#     rcm=np.dot(np.array([np.mean(X[:,0]),np.mean(X[:,1]),np.mean(X[:,2])]), axes)
#     res=0
#     N=X.shape[0]
#     for ii in range(N): res+=(np.linalg.norm(np.dot(X[ii,:], axes)-rcm))**2/N
#     return np.sqrt(res),rcm

# def calc_stiffness_traj(trajFile = 'NativeNativeTraj_0.cndb', n_arr = [1, 5, 20, 50, 100, 200]):
    
#     start=time.time()
    
#     chro = 7
#     chro1 = h5py.File('output_files/'+trajFile, 'r') # comando para abrir o arquivo

#     frames = len(chro1.keys()) - 1
#     firstkey = list(chro1.keys())[0]
    
#     print('Chromosome: Chro {:} with {:} beads'.format(chro, (len(chro1[firstkey]))))

#     Cos_Dict = {}
#     ctr = 0
    
#     flabels = np.sort(np.array(list(chro1.keys())))
#     flabels = flabels[:len(flabels)-1]
#     print(len(flabels))
#     for i, flabel in enumerate(flabels, start = 1):
        
#         if i % 500 == 0:
#             print('Analyzing Frame: ' + str(i) + ' out of ' + str(frames))
#         XYZ = np.asarray(chro1[str(flabel)])
#         if i >eqFrames:
#             ctr +=1
#             for n in n_arr:
#                 ContourLens, cos_mean, cos_std = calc_stiffness(XYZ,n)
#                 if  not (n in Cos_Dict.keys()):
#                     Cos_Dict[n] = {}
#                     Cos_Dict[n]['len'] = ContourLens
#                     Cos_Dict[n]['cos'] = cos_mean
#                 else:
#                     Cos_Dict[n]['len'] = np.add(Cos_Dict[n]['len'],ContourLens) 
#                     Cos_Dict[n]['cos'] = np.add(Cos_Dict[n]['cos'],cos_mean)
    
#     for n in n_arr:
#         Cos_Dict[n]['len'] = np.divide(Cos_Dict[n]['len'], ctr).tolist()
#         Cos_Dict[n]['cos'] = np.divide(Cos_Dict[n]['cos'],ctr).tolist()
        
#     end     = time.time()
#     elapsed = end - start

#     chro1.close()
#     chro2.close()

#     end     = time.time()
#     elapsed = end - start

#     print('Total frames: {:}'.format(i))
#     print('Ran in %.3f sec' % elapsed)
#     print('############################################################')
    
#     return Cos_Dict


# In[33]:


# Cos_Dict = calc_stiffness_traj(trajFile = 'NativeNativeTraj_0.cndb', n_arr = n_arr)


# In[34]:


# print(Cos_Dict)
# # Store data (serialize)
# f = open("output_files/CosDict.txt","w")
# f.write( str(Cos_Dict) )
# f.close()


# ## MSD of reaction coordinate: (Not Calculated Here

# In[35]:


# dt = sim.timestep*blockSize #time step between coordinate measurements.
# max_tau = 50
# max_Steps = int(max_tau/dt)
# eqBlocks = eqFrames*blocksPerFrame
# MSDs = []
# for steps in range(max_Steps):
#     MSDs.append(np.mean(np.square(np.subtract(pcoord[eqBlocks:len(pcoord)-steps], pcoord[eqBlocks+steps:]))))
    
# tau = np.linspace(0,dt*len(MSDs), len(MSDs))
# MSD_Data = np.vstack([tau, MSDs]).T
# np.savetxt('output_files/Native_MSD.txt', MSD_Data)


# In[36]:


# ##Plot MSD
# fig, ax = plt.subplots()
# ax.plot(tau, MSDs)
# ax.set_xlabel(r"Time Difference ($\tau$)")
# ax.set_ylabel(r"Mean Square Difference ($\sigma^2$)")
# ax.set_title("Diffusion of Reaction Coordinate")


# # COM Distance Reporter Object: (Not Used Here)

# In[37]:


# class CentroidForceReporter(object):
#     ''' Calculate the distance between the centroids of two groups and the current pulling force
    
#     This will calculate the mean force on particle index 1 and particle
#     index 2 along the COM-COM distance between the two particle sets.
#     Only forces in given OpenMM force is used to calculate
#     the forces, hence the manual call to `getState()`. Periodic boundaries
#     and minimum image distances are *not* considered, so make sure molecules
#     are not wrapped, nor can the COM-COM distance be longer than half the
#     box length.
    
#     Keyword arguments:
#     cm1    -- index of particle situated on mass center 1 (int)
#     cm2    -- index of particle situated on mass center 1 (int)
#     index1 -- particle index of group1 (list)
#     index2 -- particle index of group1 (list)
#     groups -- force groups to extract forces from (set)
#     reportInterval -- Steps between each sample (int)
#     '''
#     def __init__(self, pullforce, reportInterval):
        
#         self._reportInterval = reportInterval
#         self.group1 = pforce.getGroupParameters(0)[0]
#         self.group2 = pforce.getGroupParameters(1)[0]
#         self.force = 0 # pulling force (kJ/mol/nm)
#         self.r  = 0 # COM-COM distance between pull groups
#         self.bondParams = (0,0)
#         EnerEqn = pullforce.getEnergyFunction()
#         if 'kp' in EnerEqn:
#             #First entry is spring constant, second entry is reference distance.
#             self.bondParams = (pullforce.getGlobalParameterDefaultValue(0), pullforce.getGlobalParameterDefaultValue(1))
#         elif 'f' in EnerEqn:
#             self.bondParams = (pullforce.getGlobalParameterDefaultValue(0),)

#     def describeNextReport(self, simulation):
#         steps = self._reportInterval - simulation.currentStep%self._reportInterval
#         return (steps, False, False, False, False)
    
#     def clear(self):
#         '''Resets the COM distance and force'''
#         self.r=0
#         self.force=0
        
#     def meanforce(self):
#         '''Returns average meanforce along COM-COM vector (kJ/mol/nm)'''
#         if self.cnt>0:
#             return 0.5*(self.mf1+self.mf2) / self.cnt, self.mf1/self.cnt, self.mf2/self.cnt, self.mR/self.cnt
#         else:
#             print('no force samples')

#     def report(self, simulation, state):
#         s = simulation.context.getState(getPositions=True) # system
#         positions = s.getPositions(asNumpy=True).value_in_unit(nanometer) # distances
        
#         first_centroid = np.mean(positions[self.group1], axis=0) 
#         second_centroid = np.mean(positions[self.group2], axis=0)

#         ## calculate r0 distance between groups

#         r0 = np.sqrt( (first_centroid[0] - second_centroid[0])**2 + 
#                       (first_centroid[1] - second_centroid[1])**2 +
#                       (first_centroid[2] - second_centroid[2])**2 )
        
#         self.r = r0
        
#         if len(self.bondParams) == 1:
#             self.force = self.bondParams[0]
#         if len(self.bondParams) == 2:
#             self.force = self.bondParams[0]*(r0-self.bondParams[1])

