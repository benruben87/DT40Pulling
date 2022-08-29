#!/usr/bin/env python
# coding: utf-8

# # DT40 15m Open Simulation
# ## Runs Simulation for OpenMiChroM without spherical hard wall.
# 
# Initiates from oriented structure and periodically resets the center of mass to the origin to prevent large numbers.

# In[26]:


from openmichrolib import Akroma as openmichrom
from openmichrolib import SerraAngel as trainertool
import numpy as np
import time


# ## Centering Function

# In[27]:


def centerCoords(curpos):
    return curpos-(np.mean(curpos, axis=0))


# ## Simulation Parameters (To Be Changed Before Running on NOTS):

# In[28]:


platform = 'OpenCL'
blockSize = 3000
numBlocks = 20000
eqFrames = 2000 #Number of frames to skip when calculating mean energy components, contact maps, and bond cosines.
repNum = REPNUM

ortdFName = 'Ortd_15m_' + str(repNum) + '.pdb'


# In[29]:


##Start openMiChroM lib
sim = openmichrom(name='sim', temperature=120)
sim.setup(platform=platform, integrator="Langevin")


# In[30]:


#Folder to save outputs
sim.saveFolder('output_files')


# In[31]:


#read/creation of the initial state
mypol = sim.create_springSpiral(type_list='input/DT40_chr7.eigen')


# In[32]:


#Load initial conformation into the system
sim.load(mypol, center=True)
sim_aux = trainertool(TypeList=sim.type_list, state=sim.getData())


# In[33]:


#Switch conformation to input file:

sim.setData(sim.load_Coords_PDB(ortdFName))


# In[34]:


#if want save initial configuration in pdb format
sim.save(mode = 'pdb', pdbGroups=sim.chains, filename='loaded_from_ortd.pdb')


# In[35]:


#start HP forces
sim.addGrosbergPolymerBonds(k=30) 
sim.addGrosbergStiffness(k=1.0)
sim.addRepulsiveSoftCore(Et=4.0)


# In[36]:


#MiChroM potentials
#types lambdas = [AA,AB,BB]
type_lambs = sim_aux.getlambfromfile('input/types-DT40-15m')
sim.addTypes_forTrainner(mi=1.51, rc = 2.12, lambdas=type_lambs)


# In[37]:


#Ideal Chromosome potential
#REMEMBER lambs counts all iteraction, the file must have the number of line equals the size of your polymer
#the 3 first lambdas must be zero if want start interaction after 3  neighbor 
sim.lambs = sim_aux.getlambfromfile('input/lambdas_ic_15m')
sim.IC_forTrainner(mi=1.51, rc = 2.12, dcutl=3, dcutupper=735)


# In[38]:


#NO spherical Confinement
######################sim.addSphericalConfinement(density=0.1, k=0.2)


# In[39]:


#define name to save .cndb
sim.initStorage('traj', mode='w')


# In[40]:


#Matrix to save energy components:
enerComps = []


# In[41]:


#number of blocks simulation
nb=numBlocks
bs = blockSize


# In[42]:


sim.integrator.getFriction()


# In[43]:


#run simulation
time1 = time.time()
for t in range(nb+1):
    sim.doBlock(bs, increment=True, output=True) #relaxamento
    ##save trajectory
    sim.setData(centerCoords(sim.getData()));
    sim.save()
    #if want print forces
    enerNames, curEnerComps = sim.getForces()
    enerComps.append(curEnerComps)
    if t % 100 == 0:
     sim.printForces()
     print("Radius of Gyration: {:.2f}\tBlock: {}/{}".format(sim.RG(), t,nb))
     sim.save(mode = 'pdb', pdbGroups=sim.chains)
time2 = time.time()
print('This run took {:.2f} seconds'.format(time2-time1))


# In[44]:


#close storage file
sim.storage[0].close()

#save last conformation in pdb
sim.save(mode = 'pdb', pdbGroups=sim.chains)


# # Analysis
# 
# ## Energy Components

# In[45]:


import numpy as np
import time
import h5py
from   scipy.spatial  import distance


# In[46]:


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


# In[ ]:




