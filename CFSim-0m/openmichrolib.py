#!/usr/bin/env python

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

#some units
nm = units.meter * 1e-9
fs = units.second * 1e-15
ps = units.second * 1e-12


# In[25]:


class Akroma:
    
    ##initialize the akroma set##
    def __init__(
        self, timestep=0.01, thermostat=0.1, temperature=120,
        verbose=False,
        velocityReinitialize=True,
        # reinitialize velocities at every block if E_kin is more than 2.4
        name="sim",
        length_scale=1.0,
        mass_scale=1.0):  # name to print out 
            self.name = name
            self.timestep = timestep #in tau
            self.collisionRate = thermostat  #in 1/tau
            self.temperature = temperature 
            self.verbose = verbose
            self.velocityReinitialize = velocityReinitialize
            self.loaded = False  # check if the data is loaded
            self.forcesApplied = False
            self.folder = "."
            self.metadata = {}
            self.length_scale = length_scale
            self.mass_scale = mass_scale
            self.eKcritical = 50000000  # Max allowed kinetic energy
            self.nm = nm
            self.Sigma = 1.0
            self.Epsilon = 1.0
            #####################        A1         A2        B1        B2        B3        B4       NA   
            self.inter_Chrom_types =[-0.268028,-0.274604,-0.262513,-0.258880,-0.266760,-0.266760,-0.225646, #A1
                                     -0.274604,-0.299261,-0.286952,-0.281154,-0.301320,-0.301320,-0.245080, #A2
                                     -0.262513,-0.286952,-0.342020,-0.321726,-0.336630,-0.336630,-0.209919, #B1
                                     -0.258880,-0.281154,-0.321726,-0.330443,-0.329350,-0.329350,-0.282536, #B2
                                     -0.266760,-0.301320,-0.336630,-0.329350,-0.341230,-0.341230,-0.349490, #B3
                                     -0.266760,-0.301320,-0.336630,-0.329350,-0.341230,-0.341230,-0.349490, #B4
                                     -0.225646,-0.245080,-0.209919,-0.282536,-0.349490,-0.349490,-0.255994] #NA
            
    #############################################
    #important features must be initialize here!!
    #############################################
    def setup(self, platform="CUDA", PBC=False, PBCbox=None, GPU="default",
              integrator="langevin", errorTol=None, precision="double"):
        
        """Sets up the important low-level parameters of the platform.
        Mandatory to run.

        Parameters
        ----------

        platform : string, optional
            Platform to use

        PBC : bool, optional
            Use periodic boundary conditions, default:False

        PBCbox : (float,float,float), optional
            Define size of the bounding box for PBC

        GPU : "0" or "1", optional
            Switch to another GPU. Mostly unnecessary.
            Machines with 1 GPU automatically select right GPU.
            Machines with 2 GPUs select GPU that is less used.

        integrator : "langevin", "variableLangevin", "verlet", "variableVerlet",
                     "brownian", optional Integrator to use
                     (see Openmm class reference)

        verbose : bool, optional
            Shout out loud about every change.

        errorTol : float, optional
            Error tolerance parameter for variableLangevin integrator
            Values of 0.03-0.1 are reasonable for "nice" simulation
            Simulations with strong forces may need 0.01 or less

        """

        self.step = 0
        if PBC == True:
            self.metadata["PBC"] = True

        precision = precision.lower()
        if precision not in ["mixed", "single", "double"]:
            raise ValueError("Presision must be mixed, single or double")

        self.kB = units.BOLTZMANN_CONSTANT_kB *             units.AVOGADRO_CONSTANT_NA  # Boltzmann constant
        self.kT = self.kB * self.temperature  # thermal energy
        self.mass = 10.0 * units.amu * self.mass_scale
        # All masses are the same,
        # unless individual mass multipliers are specified in self.load()
        self.bondsForException = []
        self.mm = openmm
        self.conlen = 1. * nm * self.length_scale
        self.system = self.mm.System()
        self.PBC = PBC

        if self.PBC == True:  # if periodic boundary conditions
            if PBCbox is None:  # Automatically setting up PBC box
                data = self.getData()
                data -= np.min(data, axis=0)

                datasize = 1.1 * (2 + (np.max(self.getData(), axis=0) -                                        np.min(self.getData(), axis=0)))
                # size of the system plus some overhead

                self.SolventGridSize = (datasize / 1.1) - 2
                print("density is ", self.N / (datasize[0]
                    * datasize[1] * datasize[2]))
            else:
                PBCbox = np.array(PBCbox)
                datasize = PBCbox

            self.metadata["PBCbox"] = PBCbox
            self.system.setDefaultPeriodicBoxVectors([datasize[0], 0.,
                0.], [0., datasize[1], 0.], [0., 0., datasize[2]])
            self.BoxSizeReal = datasize

        self.GPU = str(GPU)  # setting default GPU
        properties = {}
        if self.GPU.lower() != "default":
            properties["DeviceIndex"] = str(GPU)
            properties["Precision"] = precision
        self.properties = properties

        if platform.lower() == "opencl":
            platformObject = self.mm.Platform.getPlatformByName('OpenCL')

        elif platform.lower() == "reference":
            platformObject = self.mm.Platform.getPlatformByName('Reference')

        elif platform.lower() == "cuda":
            platformObject = self.mm.Platform.getPlatformByName('CUDA')

        elif platform.lower() == "cpu":
            platformObject = self.mm.Platform.getPlatformByName('CPU')


        else:
            self.exit("\n!!!!!!!!!!unknown platform!!!!!!!!!!!!!!!\n")
        self.platform = platformObject

        self.forceDict = {}  # Dictionary to store forces

        self.integrator_type = integrator
        if isinstance(integrator, string_types):
            integrator = str(integrator)
            if integrator.lower() == "langevin":
                self.integrator = self.mm.LangevinIntegrator(self.temperature,
                    self.collisionRate, self.timestep)
            elif integrator.lower() == "variablelangevin":
                self.integrator = self.mm.VariableLangevinIntegrator(self.temperature,
                    self.collisionRate, errorTol)
            elif integrator.lower() == "verlet":
                self.integrator = self.mm.VariableVerletIntegrator(self.timestep)
            elif integrator.lower() == "variableverlet":
                self.integrator = self.mm.VariableVerletIntegrator(errorTol)

            elif integrator.lower() == 'brownian':
                self.integrator = self.mm.BrownianIntegrator(self.temperature,
                    self.collisionRate, self.timestep)
            else:
                print ('please select from "langevin", "variablelangevin", '
                       '"verlet", "variableVerlet", '
                       '"brownian" or provide an integrator object')
        else:
            self.integrator = integrator
            self.integrator_type = "UserDefined"
    
    ###################################################
    ###### Folder to  save your simulation ############
    ###################################################
    def saveFolder(self, folder):
        """
        sets the folder where to save data.

        Parameters
        ----------
            folder : string
                folder to save the data

        """
        if os.path.exists(folder) == False:
            os.mkdir(folder)
        self.folder = folder

    ##########################################################
    ##### load you molecule in openmm, this is important! ####
    ##########################################################
    def load(self, filename,  # input data array
             center=False,  # Shift center of mass to zero?
             masses=None,
             ):

        """
        center : bool, optional
            Move center of mass to zero before starting the simulation
        masses : array
            Masses of each atom, measured in self.mass (default: 100 AMU,
            but could be modified by self.mass_scale)
        """
        data = filename

        data = np.asarray(data, float)

        if len(data) == 3:
            data = np.transpose(data)
        if len(data[0]) != 3:
            self._exitProgram("strange data file")
        if np.isnan(data).any():
            self._exitProgram("\n!!!!!!!!!!file contains NANS!!!!!!!!!\n")

        if center is True:
            av = np.mean(data, 0)
            data -= av

        if center == "zero":
            minvalue = np.min(data, 0)
            data -= minvalue

        self.setData(data) #import data to openmm
        #self.randomizeData() #just a randomize in u use intergers

        if masses == None:
            self.masses = [1. for _ in range(self.N)]
        else:
            self.masses = masses
            
        if not hasattr(self, "chains"):
            self.setChains()
            
    def setChains(self, chains=[(0, None, 0)]):
        """
        Sets configuration of the chains in the system. This information is
        later used by the chain-forming methods, e.g. addHarmonicPolymerBonds()
        and addStiffness().
        This method supersedes the less general getLayout().

        Parameters
        ----------

        chains: list of tuples
            The list of chains in format [(start, end, isRing)]. The particle
            range should be semi-open, i.e. a chain (0,3,0) links
            the particles 0, 1 and 2. If bool(isRing) is True than the first
            and the last particles of the chain are linked into a ring.
            The default value links all particles of the system into one chain.
        """

        #f not hasattr(self, "N"):
        #   raise ValueError("Load the chain first, or provide chain length")

        self.chains = [i for i in chains]  # copy
        for i in range(len(self.chains)):
            start, end, isRing = self.chains[i]
            #end = self.N if (end is None) else end
            self.chains[i] = (start, end, isRing)

    def setData(self, data, random_offset = 1e-5):
        """Sets particle positions

        Parameters
        ----------

        data : Nx3 array-line
            Array of positions with distance ~1 between connected atoms.
        """
        data = np.asarray(data, dtype="float")
        if random_offset:
            data = data + (np.random.random(data.shape) * 2 - 1) * random_offset
        
        self.data = units.Quantity(data, nm)
        self.N = len(self.data)
        if hasattr(self, "context"):
            self.initPositions()
            

    def randomizeData(self):
        """
        Runs automatically to offset data if it is integer based
        """
        data = self.getData()
        data = data + np.random.randn(*data.shape) * 0.0001
        self.setData(data)

    def getData(self):
        "Returns an Nx3 array of positions"
        return np.asarray(self.data / nm, dtype=np.float32)
    
    def getLoops(self, filenames):
        self.loopPosition = []
        for file, chain in zip(filenames,self.chains):
            aFile = open(file,'r')
            pos = aFile.read().splitlines()
            m = int(chain[0])
            #print(m)
            for t in range(len(pos)):
                pos[t] = pos[t].split()
                pos[t][0] = int(pos[t][0]) +m
                pos[t][1] = int(pos[t][1]) +m
                self.loopPosition.append(pos[t])
        

        
    ##########################################################################################
        ################## FORCES AND BONDS HERE ########################################
    ##########################################################################################

    def addSphericalConfinement(self,
                r="density",  # radius... by default uses certain density
                k=5.,  # How steep the walls are
                density=0.1):  # target density, measured in particles
                                # per cubic nanometer (bond size is 1 nm)
        """Constrain particles to be within a sphere.
        With no parameters creates sphere with density .3

        Parameters
        ----------
        r : float or "density", optional
            Radius of confining sphere. If "density" requires density,
            or assumes density = .3
        k : float, optional
            Steepness of the confining potential, in kT/nm
        density : float, optional, <1
            Density for autodetection of confining radius.
            Density is calculated in particles per nm^3,
            i.e. at density 1 each sphere has a 1x1x1 cube.
        """
        self.metadata["SphericalConfinement"] = repr({"r": r, "k": k,
            "density": density})

        spherForce = self.mm.CustomExternalForce(
            "step(r-SPHaa) * SPHkb * (sqrt((r-SPHaa)*(r-SPHaa) + SPHt*SPHt) - SPHt) "
            ";r = sqrt(x^2 + y^2 + z^2 + SPHtt^2)")

        for i in range(self.N):
            spherForce.addParticle(i, [])
        if r == "density":
            r = (3 * self.N / (4 * 3.141592 * density)) ** (1 / 3.)

        self.sphericalConfinementRadius = r
        if self.verbose == True:
            print("Spherical confinement with radius = %lf" % r)
        # assigning parameters of the force
        spherForce.addGlobalParameter("SPHkb", k )
        spherForce.addGlobalParameter("SPHaa", (r - 1. / k) )
        spherForce.addGlobalParameter("SPHt", (1. / k) / 10.)
        spherForce.addGlobalParameter("SPHtt", 0.01)
        
        self.forceDict["SphericalConfinement"] = spherForce
        
        return r
    
    
    def addSphericalConfinementLJ(self,
                r="density",  # radius... by default uses certain density
                density=0.1):  # target density, measured in particles
                                # per cubic nanometer (bond size is 1 nm)
 
        spherForce = self.mm.CustomExternalForce("(4 * GROSe * ((GROSs/r)^12 - (GROSs/r)^6) + GROSe) * step(GROScut - r);"
                                                 "r= R - sqrt(x^2 + y^2 + z^2) ")
            

        for i in range(self.N):
            spherForce.addParticle(i, [])
        if r == "density":
            r = (3 * self.N / (4 * 3.141592 * density)) ** (1 / 3.)

        self.sphericalConfinementRadius = r

        # assigning parameters of the force
        spherForce.addGlobalParameter('R', r)
        spherForce.addGlobalParameter('GROSe', 1.0)
        spherForce.addGlobalParameter('GROSs', 1.0)
        spherForce.addGlobalParameter("GROScut", 2.**(1./6.))
        
        self.forceDict["SphericalConfinementLJ"] = spherForce
        
        return r
    
    def addHarmonicPolymerBonds(self,
                                wiggleDist=0.05,
                                bondLength=1.0,
                                exceptBonds=True):
        """Adds harmonic bonds connecting polymer chains

        Parameters
        ----------

        wiggleDist : float
            Average displacement from the equilibrium bond distance
        bondLength : float
            The length of the bond
        exceptBonds : bool
            If True then do not calculate non-bonded forces between the
            particles connected by a bond. True by default.
        """


        for start, end, isRing in self.chains:
            for j in range(start, end - 1):
                self.addBond(j, j + 1, wiggleDist,
                    distance=bondLength,
                    bondType="Harmonic", verbose=False)
                if exceptBonds:
                    self.bondsForException.append((j, j + 1))

            if isRing:
                self.addBond(start, end - 1, wiggleDist,
                    distance=bondLength, bondType="Harmonic")
                if exceptBonds:
                    self.bondsForException.append((start, end - 1))
                if self.verbose == True:
                    print("ring bond added", start, end - 1)

        self.metadata["HarmonicPolymerBonds"] = repr(
            {"wiggleDist": wiggleDist, 'bondLength':bondLength})
   
    def _initHarmonicBondForce(self):
        "Internal, inits harmonic forse for polymer and non-polymer bonds"
        if "HarmonicBondForce" not in list(self.forceDict.keys()):
            self.forceDict["HarmonicBondForce"] = self.mm.HarmonicBondForce()
        self.bondType = "Harmonic"
        
        
    def addGrosbergPolymerBonds(self, k=30):
        """Adds FENE bonds according to Halverson-Grosberg paper.
        (Halverson, Jonathan D., et al. "Molecular dynamics simulation study of
         nonconcatenated ring polymers in a melt. I. Statics."
         The Journal of chemical physics 134 (2011): 204904.)

        This method has a repulsive potential build-in,
        so that Grosberg bonds could be used with truncated potentials.
        Is of no use unless you really need to simulate Grosberg-type system.

        Parameters
        ----------
        k : float, optional
            Arbitrary parameter; default value as in Grosberg paper.

         """

        for start, end, isRing in self.chains:
            for j in range(start, end):
                self.addBond(j, j + 1, bondType="Grosberg")
                self.bondsForException.append((j, j + 1))

            if isRing:
                self.addBond(start, end - 1, distance=1, bondType="Harmonic")
                self.bondsForException.append((start, end ))
                if self.verbose == True:
                    print("ring bond added", start, end )

        self.metadata["GorsbergPolymerForce"] = repr({"k": k})
        
    def _initGrosbergBondForce(self):
        "inits Grosberg FENE bond force"
        if "GrosbergBondForce" not in list(self.forceDict.keys()):
            #force = ("- 0.5 * GROSk * GROSr0 * GROSr0 * log(1-(r/GROSr0)*(r/GROSr0)) + (4 * GROSe * ((GROSs/r)^12 - (GROSs/r)^6) + GROSe) * step(GROScut - r)")
            #force = (" (GROSk*GROSrmax*(1-(GROSrmax/GROSr0)^2)^-1 * (r-GROSrmax)+(- 0.5 * GROSk * GROSr0 * GROSr0 * log(1-(GROSrmax/GROSr0)*(GROSrmax/GROSr0)))) * step(r-GROSrmax)+  (- 0.5 * GROSk * GROSr0 * GROSr0 * log(1-(r/GROSr0)*(r/GROSr0)))*step(GROSrmax-r) + (4 * GROSe * ((GROSs/r)^12 - (GROSs/r)^6) + GROSe) * step(GROScut - r)")
            force = (" (GROSk * (r-GROSrmax)+(- 0.5 * GROSk * GROSr0 * GROSr0 * log(1-(GROSrmax/GROSr0)*(GROSrmax/GROSr0)))) * step(r-GROSrmax)+  (- 0.5 * GROSk * GROSr0 * GROSr0 * log(1-(r/GROSr0)*(r/GROSr0)))*step(GROSrmax-r) + (4 * GROSe * ((GROSs/r)^12 - (GROSs/r)^6) + GROSe) * step(GROScut - r)") #Uses GROSk as the constant extension force.  Softer than a continuous extension to prevent blow-up.
            bondforceGr = self.mm.CustomBondForce(force)
            bondforceGr.addGlobalParameter("GROSrmax", 1.425)# Distance cutoff for divergence of grosberg bond force.  Avoide huge force values and chain breakage during energy minimization.  Added by Ben Ruben.
            bondforceGr.addGlobalParameter("GROSk", 30)# *
                #self.kT / (self.conlen * self.conlen))
            bondforceGr.addGlobalParameter("GROSr0", 1.5) #self.conlen * 1.5)
            bondforceGr.addGlobalParameter('GROSe', 1.0)#self.kT)
            bondforceGr.addGlobalParameter('GROSs', 1.0)#self.conlen)
            bondforceGr.addGlobalParameter("GROScut", 2.**(1./6.))
                #"GROScut", self.conlen * 2. ** (1. / 6.))
            self.forceDict["GrosbergBondForce"] = bondforceGr

    def addBond(self,
                i, j,  # particles connected by bond
                bondWiggleDistance=0.2,
                # Flexibility of the bond,
                # measured in distance at which energy equals kT
                distance=None,  # Equilibrium length of the bond
                bondType=None,  # Harmonic, Grosberg, ABS
                verbose=None):  # Set this to False if you're in verbose mode
                # and don't want to contaminate output by 10000 messages
        """Adds bond between two particles, allows to specify parameters

        Parameters
        ----------

        i,j : int
            Particle numbers

        bondWiggleDistance : float
            Average displacement from the equilibrium bond distance

        bondType : "Harmonic" or "Grosberg"
            Type of bond. Distance and bondWiggleDistance can be
            specified for harmonic bonds only

        verbose : bool
            Set this to False if you're in verbose mode and don't want to
            print "bond added" message

        """

        if not hasattr(self, "bondLengths"):
            self.bondLengths = []

        if verbose is None:
            verbose = self.verbose
        if (i >= self.N) or (j >= self.N):
            raise ValueError("\nCannot add bond with monomers %d,%d that"            "are beyound the polymer length %d" % (i, j, self.N))
        bondSize = float(bondWiggleDistance)
        if distance is None:
            distance = self.length_scale
        else:
            distance = self.length_scale * distance
        distance = float(distance)

        #if not hasattr(self, "kbondScalingFactor"):
        #    self.kbondScalingFactor = float((2 * self.kT / (self.conlen) ** 2) / (units.kilojoule_per_mole / nm ** 2))
        #kbondScalingFactor = self.kbondScalingFactor


        if bondType is None:
            bondType = self.bondType

        if bondType.lower() == "harmonic":
            self._initHarmonicBondForce()
            kbond = kbondScalingFactor / (bondSize ** 2)
            self.forceDict["HarmonicBondForce"].addBond(int(i), int(j), float(distance), float(kbond))
            self.bondLengths.append([int(i), int(j), float(distance), float(bondSize)])
        elif bondType.lower() == "grosberg":
            self._initGrosbergBondForce()
            self.forceDict["GrosbergBondForce"].addBond(int(i), int(j), [])
        elif bondType.lower() == "abs":
            self._initAbsBondForce()
            self.forceDict["AbsBondForce"].addBond(int(i), int(
                j), [float(bondWiggleDistance), float(distance)])
            self.bondLengths.append([int(i), int(j), float(distance), float(bondSize)])
        elif bondType.lower() == "abslim":
            self._initAbsDistanceLimitation()
            self.forceDict["AbsLimitation"].addBond(int(i), int(
                j), [float(bondWiggleDistance), float(distance)])

        else:
            self._exitProgram("Bond type not known")
        if verbose == True:
            print("%s bond added between %d,%d, wiggle %lf dist %lf" % (
                bondType, i, j, float(bondWiggleDistance), float(distance)))
    
    def addGrosbergStiffness(self, k=1.0):
        """Adds stiffness according to the Grosberg paper.
        (Halverson, Jonathan D., et al. "Molecular dynamics simulation study of
         nonconcatenated ring polymers in a melt. I. Statics."
         The Journal of chemical physics 134 (2011): 204904.)

        Parameters are synchronized with normal stiffness

        If k is an array, it has to be of the length N.
        Xth value then specifies stiffness of the angle centered at
        monomer number X.
        Values for ends of the chain will be simply ignored.

        Parameters
        ----------

        k : float or N-long list of floats
            Synchronized with regular stiffness.
            Default value is very flexible, as in Grosberg paper.
            Default value maximizes entanglement length.

        """
        try:
            k[0]
        except:
            k = np.zeros(self.N, float) + k
        stiffForce = self.mm.CustomAngleForce(
            "GRk *  (1 - cos(theta - 3.141592))")
        

        #stiffForce.addGlobalParameter("kT", self.kT)
        stiffForce.addPerAngleParameter("GRk")
        for start, end, isRing in self.chains:
            for j in range(start + 1, end):
                stiffForce.addAngle(j - 1, j, j + 1, [k[j]])
            if isRing:
                stiffForce.addAngle(end - 1, end , start, [k[end]])
                stiffForce.addAngle(end , start, start + 1, [k[start]])

        self.metadata["GrosbergAngleForce"] = repr({"stiffness": k})
        self.forceDict["AngleForce"] = stiffForce
    
    def addRepulsiveSoftCore(self, Et=4.0):
       
        
        nbCutOffDist = self.Sigma * 2. ** (1. / 6.) # cutoff in sigma*2^(1/6) #1.112
        
        Ecut = Et*self.Epsilon #Ecut Definition
        # NOTe: must be changed if E_cut value is changed
        r_0 = self.Sigma * 2**(1/6.0) * (-1 + 2**(1/2.0) )**(1/6.0) #0.96
        
        #" + step(r_0 - r)* 0.5 * Ecut * (1.0 + tanh( (2.0 * LJ/Ecut) - 1.0 ));"
        #"LJ = 4.0 * Epsi * ((Sig/r)^12 - (Sig/r)^6) + Epsi"
        #* step(r - r_0) * step(CutOff - r)
        repul_energy = ("LJ * step(r - r_0) * step(CutOff - r)"
                       " + step(r_0 - r)* 0.5 * Ecut * (1.0 + tanh( (2.0 * LJ/Ecut) - 1.0 ));"
                       "LJ = 4.0 * Epsi * ((Sig/r)^12 - (Sig/r)^6) + Epsi")
        
        
        repulforceGr = self.mm.CustomNonbondedForce(repul_energy)
        repulforceGr.addGlobalParameter('Epsi', self.Epsilon)
        repulforceGr.addGlobalParameter('Sig', self.Sigma)
        repulforceGr.addGlobalParameter('Ecut', Ecut)
        repulforceGr.addGlobalParameter('r_0', r_0)
        repulforceGr.addGlobalParameter('CutOff', nbCutOffDist)
        repulforceGr.setCutoffDistance(3.0)

        for _ in range(self.N):
            repulforceGr.addParticle(())
            
        self.forceDict["RepulsiveSoftCore"] = repulforceGr
        
    def addCrossLinkProb(self, mi=3.22, rc = 1.78 ):
        '''this force is based in type to type interactions, important see the list of types first!
        '''

        self.metadata["CrossLink"] = repr({"mi": mi})
        if not hasattr(self, "type_list"): #if any list exist, create a random one!
             self.type_list = self.random_type(self.N)

        energy = "mapType(t1,t2)*0.5*(1. + tanh(mi*(rc - r)))*step(r-1.0)"
        
        crossLP = self.mm.CustomNonbondedForce(energy)
    
        crossLP.addGlobalParameter('mi', mi)
        crossLP.addGlobalParameter('rc', rc)
        crossLP.setCutoffDistance(3.0)
        
        fTypes = self.mm.Discrete2DFunction(7,7,self.inter_Chrom_types) #create de tabular function
        crossLP.addTabulatedFunction('mapType', fTypes)  ##add tabular function
        
        #print(fTypes.getFunctionParameters())
        
        crossLP.addPerParticleParameter("t")

        for i in range(self.N):
                value = [float(self.type_list[i])]
                crossLP.addParticle(value)
                
                
        self.forceDict["CrossLink"] = crossLP 
        
    def addTypes_forTrainner(self, mi=3.22, rc = 1.78, lambdas=[0,0,0]):
        '''this force is based in type to type interactions, important see the list of types first!
        '''

        self.metadata["CrossLink"] = repr({"mi": mi})
        if not hasattr(self, "type_list"): #if any list exist, create a random one!
             self.type_list = self.random_type(self.N)

        energy = "mapType(t1,t2)*0.5*(1. + tanh(mi*(rc - r)))*step(r-1.0)"
        
        crossLP = self.mm.CustomNonbondedForce(energy)
    
        crossLP.addGlobalParameter('mi', mi)
        crossLP.addGlobalParameter('rc', rc)
        crossLP.setCutoffDistance(3.0)
        
        #                    AA AB BB
        #REMEMBER lambdas = [0, 0, 0]
        #
        
        lambdas = [lambdas[0],lambdas[1], lambdas[1],lambdas[2]]      
        
        fTypes = self.mm.Discrete2DFunction(2,2,lambdas) #create de tabular function
        crossLP.addTabulatedFunction('mapType', fTypes)  ##add tabular function
        
        #print(fTypes.getFunctionParameters())
        
        AB_types = [1 if value == 2 else value for value in self.type_list]
        crossLP.addPerParticleParameter("t")

        for i in range(self.N):
                value = [float(AB_types[i])]
                crossLP.addParticle(value)
                
        self.forceDict["CrossLink"] = crossLP

#    def addLoopSites(self, mi=3.22, rc = 1.78, X=-1.612990, filename=None):
#        '''
#        Loops site interaction
#        '''
#
#        ELoop = "Fwork(a1,a2)*qsi*0.5*(1. + tanh(mi*(rc - r)))"
#
#        Loop = self.mm.CustomNonbondedForce(ELoop)
       
#        Loop.addGlobalParameter('mi', mi)  
#        Loop.addGlobalParameter('rc', rc) 
#        Loop.addGlobalParameter('qsi', X) 
#        Loop.setCutoffDistance(3.0)
        
#        self.getLoops(filename)
        
        #for loop in self.loopPosition:
        #    Loop.addInteractionGroup([loop[0]-1],[loop[1]-1])
        
        #for _ in range(self.N):
        #        Loop.addParticle()
        
#        pos = np.zeros((self.N,self.N))
        
#        for p in self.loopPosition:
#            pos[(p[0]-1),(p[1]-1)] = 1.0
        
        #for i in range(1,self.N+1):
        #    for j in range(1,self.N+1):
        #        if [i,j] in self.loopPosition:
        #            pos.append(1.0)
        #         else:
        #           pos.append(0)
        #pos.append([1.0,1.0,1.0,1.0])
                    
#        work = self.mm.Discrete2DFunction(self.N,self.N, pos.ravel()) #create de tabular function
#        Loop.addTabulatedFunction('Fwork', work)  ##add tabular function
        
#        Loop.addPerParticleParameter("a")
        
#        self.index = list(range(self.N))
        
#        for i in range(self.N):
#                value = [float(self.index[i])]
#                Loop.addParticle(value)
        
#        self.forceDict["LoopSites"] = Loop  
 
    def addLoopSites(self, mi=3.22, rc = 1.78, X=-1.612990, filename=None):
        '''
        Loops site interaction
        '''

        ELoop = "qsi*0.5*(1. + tanh(mi*(rc - r)))*step(r-1.0)"
                
        Loop = self.mm.CustomBondForce(ELoop)
        
        Loop.addGlobalParameter('mi', mi)  
        Loop.addGlobalParameter('rc', rc) 
        Loop.addGlobalParameter('qsi', X) 
        
        self.getLoops(filename)
        
        for p in self.loopPosition:
            Loop.addBond(p[0]-1,p[1]-1)
  
        
        self.forceDict["LoopSites"] = Loop  

        
    def IC_forTrainner(self, mi=3.22, rc = 1.78, dcutl=3, dcutupper=203):
        '''
        ideal cromossome for training
        '''

        if not hasattr(self, "lambs"):
            self.lambs = np.zeros(self.N)
            print("RANDOM")
             
            
        
        energyIC = ("step(d-dcutlower)*lambdas(d)*step(dcutupper -d)*f*step(r-1.0);"
                    "f=0.5*(1. + tanh(mi*(rc - r)));"
                    "d=abs(idx1-idx2)")

        IC = self.mm.CustomNonbondedForce(energyIC)

        tablamb = self.mm.Discrete1DFunction(self.lambs) #create de tabular function
        IC.addTabulatedFunction('lambdas', tablamb)  ##add tabular function

        IC.addGlobalParameter('dcutlower', dcutl)  #initial cutoff d > 3
        IC.addGlobalParameter('dcutupper', dcutupper) #mode distance cutoff  d < 200
        
        IC.addGlobalParameter('mi', mi)  
        IC.addGlobalParameter('rc', rc) 
        
        IC.setCutoffDistance(3.0)


        IC.addPerParticleParameter("idx")

        for i in range(self.N):
                IC.addParticle([i])
        
        self.forceDict["IdealChromosome"] = IC
        
    def addIdealChromosome(self, mi=3.22, rc = 1.78, Gamma1=-0.030,Gamma2=-0.351,
                           Gamma3=-3.727, dcutl=3, dcutupper=500):
        '''
        ideal cromossome
        '''

        energyIC = ("step(d-dcutlower)*(gamma1/log(d) + gamma2/d + gamma3/d^2)*step(dcutupper -d)*f*step(r-1.0);"
                   "f=0.5*(1. + tanh(mi*(rc - r)));"
                   "d=abs(idx1-idx2)")

        IC = self.mm.CustomNonbondedForce(energyIC)

        IC.addGlobalParameter('gamma1', Gamma1) 
        IC.addGlobalParameter('gamma2', Gamma2)
        IC.addGlobalParameter('gamma3', Gamma3)
        IC.addGlobalParameter('dcutlower', dcutl)  #initial cutoff d > 3
        IC.addGlobalParameter('dcutupper', dcutupper) #mode distance cutoff  d < 200
        
        IC.addGlobalParameter('mi', mi)  
        IC.addGlobalParameter('rc', rc) 
        
        IC.setCutoffDistance(3.0)


        IC.addPerParticleParameter("idx")

        for i in range(self.N):
                IC.addParticle([i])
        
        self.forceDict["IdealChromosome"] = IC
        
        
    def addPoliIdealChromosome(self, mi=3.22, rc = 1.78, Gamma1=-0.030,Gamma2=-0.351,
                           Gamma3=-3.727, dcutl=3, dcutupper=200, chain=None):
        '''
        ideal cromossome
        '''

        energyIC = ("step(d-dcutlower)*(gamma1/log(d) + gamma2/d + gamma3/d^2)*step(dcutupper-d)*f*step(r-1.0);"
                   "f=0.5*(1. + tanh(mi*(rc - r)));"
                   "d=abs(idx1-idx2)")
        
        
        IC = self.mm.CustomNonbondedForce(energyIC)

        IC.addGlobalParameter('gamma1', Gamma1) 
        IC.addGlobalParameter('gamma2', Gamma2)
        IC.addGlobalParameter('gamma3', Gamma3)
        IC.addGlobalParameter('dcutlower', dcutl)  #initial cutoff d > 3
        IC.addGlobalParameter('dcutupper', dcutupper) #mode distance cutoff  d < 200
        
        #IC.addGlobalParameter('chainI', chain[0])
        #IC.addGlobalParameter('chainF', chain[1]-1)
        
        #"g=step(idx1-chainI)*step(chainF-idx2)*step(idx2-chainI)*step(chainF-idx1);"
        
        IC.addGlobalParameter('mi', mi)  
        IC.addGlobalParameter('rc', rc) 
        
        IC.setCutoffDistance(3)
        
        groupList = list(range(chain[0],chain[1]+1))
        
        #print("Ideal Cromossome apply: [", chain[0], "-", chain[1],"]")
        
        IC.addInteractionGroup(groupList,groupList)
        
        IC.addPerParticleParameter("idx")

        for i in range(self.N):
                IC.addParticle([i])
        
        self.forceDict["IdealChromosome_"+str(chain[0])] = IC
    
    def _loadParticles(self):
        if not hasattr(self, "system"):
            return
        if not self.loaded:
            for mass in self.masses:
                self.system.addParticle(self.mass * mass)
            if self.verbose == True:
                print("%d particles loaded" % self.N)
            self.loaded = True

    def _applyForces(self):
        """Adds all particles to the system.
        Then applies all the forces in the forcedict.
        Forces should not be modified after that, unless you do it carefully
        (see openmm reference)."""

        if self.forcesApplied == True:
            return
        self._loadParticles()

        exc = self.bondsForException
        print("Number of exceptions:", len(exc))

        if len(exc) > 0:
            exc = np.array(exc)
            exc = np.sort(exc, axis=1)
            exc = [tuple(i) for i in exc]
            exc = list(set(exc))  # only unique pairs are left

        for i in list(self.forceDict.keys()):  # Adding exceptions
            force = self.forceDict[i]
            if hasattr(force, "addException"):
                print('Add exceptions for {0} force'.format(i))
                for pair in exc:
                    force.addException(int(pair[0]),
                        int(pair[1]), 0, 0, 0, True)
            elif hasattr(force, "addExclusion"):
                print('Add exclusions for {0} force'.format(i))
                for pair in exc:
                    # force.addExclusion(*pair)
                    force.addExclusion(int(pair[0]), int(pair[1]))

            if hasattr(force, "CutoffNonPeriodic") and hasattr(
                                                    force, "CutoffPeriodic"):
                if self.PBC:
                    force.setNonbondedMethod(force.CutoffPeriodic)
                    print("Using periodic boundary conditions!!!!")
                else:
                    force.setNonbondedMethod(force.CutoffNonPeriodic)
            print("adding force ", i, self.system.addForce(self.forceDict[i]))

                #Create force group for each added force
        for i,name in enumerate(self.forceDict):
            self.forceDict[name].setForceGroup(i) 
        
        self.context = self.mm.Context(self.system, self.integrator, self.platform, self.properties)
        self.initPositions()
        self.initVelocities()
        self.forcesApplied = True
        




##############################################################################################                    
    ####################### FUNCTIONS FOR CREATE A POLYMER #########################
##############################################################################################

    def create_random_walk(self, step_size, N, segment_length=1):
        theta = np.repeat(np.random.uniform(0., 1., N // segment_length + 1),
                      segment_length)
        theta = 2.0 * np.pi * theta[:N]
        u = np.repeat(np.random.uniform(0., 1., N // segment_length + 1),
                  segment_length)
        u = 2.0 * u[:N] - 1.0
        x = step_size * np.sqrt(1. - u * u) * np.cos(theta)
        y = step_size * np.sqrt(1. - u * u) * np.sin(theta)
        z = step_size * u
        x, y, z = np.cumsum(x), np.cumsum(y), np.cumsum(z)
        return np.vstack([x, y, z]).T
    
    def create_spiral(self, r1, r2, N):
        """
        Creates a "propagating spiral", often used as a starting conformation.
        Run it with r1=10, r2 = 13, N=5000, and see what it does.
        """
        Pi = np.pi
        points = []
        finished = [False]

        def rad(phi):
            return phi / (2 * Pi)

        def ang(rad):
            return 2 * Pi * rad

        def coord(phi):
            r = rad(phi)
            return (r * np.sin(phi), r * np.cos(phi))

        def fullcoord(phi, z):
            c = coord(phi)
            return [c[0], c[1], z]
    
        def dist(phi1, phi2):
            c1 = coord(phi1)
            c2 = coord(phi2)
            d = np.sqrt((c1[1] - c2[1]) ** 2 + (c1[0] - c2[0]) ** 2)
            return d
    
        def nextphi(phi):
            phi1 = phi
            phi2 = phi + 0.7 * Pi
            mid = phi2
            while abs(dist(phi, mid) - 1) > 0.00001:
                mid = (phi1 + phi2) / 2.
                if dist(phi, mid) > 1:
                    phi2 = mid
                else:
                    phi1 = mid
            return mid
    
        def prevphi(phi):
    
            phi1 = phi
            phi2 = phi - 0.7 * Pi
            mid = phi2
    
            while abs(dist(phi, mid) - 1) > 0.00001:
                mid = (phi1 + phi2) / 2.
                if dist(phi, mid) > 1:
                    phi2 = mid
                else:
                    phi1 = mid
            return mid
    
        def add_point(point, points=points, finished=finished):
            if (len(points) == N) or (finished[0] == True):
                points = np.array(points)
                finished[0] = True
            else:
                points.append(point)
    
        z = 0
        forward = True
        curphi = ang(r1)
        add_point(fullcoord(curphi, z))
        while True:
            if finished[0] == True:
                return points
            if forward == True:
                curphi = nextphi(curphi)
                add_point(fullcoord(curphi, z))
                if (rad(curphi) > r2):
                    forward = False
                    z += 1
                    add_point(fullcoord(curphi, z))
            else:
                curphi = prevphi(curphi)
                add_point(fullcoord(curphi, z))
                if (rad(curphi) < r1):
                    forward = True
                    z += 1

    def loadGro(self, filename=['/home/antonio/Dropbox/Multichain/matheus/free/out21_1_L.gro']):
        #Type_conversion = {'ZA':0, 'OA':1, 'FB':2, 'SB':3,'TB':4, 'LB' :5, 'UN' :6}
        Type_conversion = {'ASP':0, 'HIS':2}
        x = []
        y = []
        z = []
        index = []
        start = 0
        chains = []
        sizeChain = 0
        
        for gro in filename:
            aFile = open(gro,'r')
            pos = aFile.read().splitlines()
            size = int(pos[1])
            #print(size)
            for t in range(2, len(pos)-1):
                pos[t] = pos[t].split()
                x.append(float(pos[t][3]))
                y.append(float(pos[t][4]))
                z.append(float(pos[t][5]))
                index.append(Type_conversion[pos[t][1]])
                sizeChain += 1


            chains.append((start, sizeChain-1, 0))
            start = sizeChain 
            
        print("cadeias: ", chains)    
        self.type_list = index
        self.index = list(range(len(self.type_list)))
        self.setChains(chains)
        return np.vstack([x,y,z]).T
                    
    def loadMultiplePDB(self, filename):
        #Type_conversion = {'ALA':0, 'ARG':1, 'ASP':2, 'GLU':3,'GLY':4, 'LEU' :5, 'ASN' :6}
        Type_conversion = {'ASP':0, 'HIS':2}
        x = []
        y = []
        z = []
        index = []
        start = 0
        chains = []
        sizeChain = 0
        
        for pdb in filename:
            aFile = open(pdb,'r')
            pos = aFile.read().splitlines()

            for t in range(len(pos)):
                pos[t] = pos[t].split()
                if pos[t][0] == 'ATOM':
                    x.append(float(pos[t][5]))
                    y.append(float(pos[t][6]))
                    z.append(float(pos[t][7]))
                    index.append(Type_conversion[pos[t][3]])
                    sizeChain += 1


            chains.append((start, sizeChain, 0))
            start = sizeChain 
            
        print("cadeias: ", chains)    
        self.type_list = index
        self.index = list(range(len(self.type_list)))
        self.setChains(chains)
        return np.vstack([x,y,z]).T
    
    def loadPDB(self, filename):#, type_list):
        print('NEW')
        aFile = open(filename,'r')
        pos = aFile.read().splitlines()
        #Type_conversion = {'ALA':0, 'ARG':1, 'ASP':2, 'GLU':3,'GLY':4, 'LEU' :5, 'ASN' :6}
        Type_conversion = {'ASP':0, 'HIS':2}
        x = []
        y = []
        z = []
        index = []

        for t in range(len(pos)):
            pos[t] = pos[t].split()
            if pos[t][0] == 'ATOM':
                x.append(float(pos[t][5]))
                y.append(float(pos[t][6]))
                z.append(float(pos[t][7]))
                index.append(Type_conversion[pos[t][3]])


        self.type_list = index
        self.index = list(range(len(self.type_list)))
        return np.vstack([x,y,z]).T
    
    def load_Coords_PDB(self, filename):#, type_list):
        print('Coords Only')
        aFile = open(filename,'r')
        pos = aFile.read().splitlines()
        #Type_conversion = {'ALA':0, 'ARG':1, 'ASP':2, 'GLU':3,'GLY':4, 'LEU' :5, 'ASN' :6}
        #Type_conversion = {'ASP':0, 'HIS':2}
        x = []
        y = []
        z = []
        #index = []

        for t in range(len(pos)):
            pos[t] = pos[t].split()
            if pos[t][0] == 'ATOM':
                x.append(float(pos[t][5]))
                y.append(float(pos[t][6]))
                z.append(float(pos[t][7]))
                #index.append(Type_conversion[pos[t][3]])


        #self.type_list = index
        #self.index = list(range(len(self.type_list)))
        return np.vstack([x,y,z]).T

                   
    def create_springSpiral(self,N=3000, type_list=None):
        x = []
        y = []
        z = []
        if not hasattr(self, "type_list"):
            self.type_list = []
        if type_list == None:
            beads = N
            self.type_list = self.random_type(beads)
        else:
            self.translate_type(type_list)
            beads = len(self.type_list)
        
        self.index = list(range(beads))    
        for i in range(beads):
            a = 1.9*np.pi*(i-1)/(beads-1)
            a1 = 1.9*np.pi*(2-1)/(beads-1)    
            b=1/np.sqrt((4-3.0*np.cos(a1)-np.cos(10*a1)*np.cos(a1))**2 +
                (0-3.0*np.sin(a1)-np.cos(10*a1)*np.sin(a1))**2+(np.sin(10*a1))**2)

            x.append(1.5*np.pi*b+3*b*np.cos(a)+b*np.cos(10*a)*np.cos(a))
            y.append(1.5*np.pi*b+3.0*b*np.sin(a)+b*np.cos(10*a)*np.sin(a))
            z.append(1.5*np.pi*b+b*np.sin(10*a))
        
        chain = []
        chain.append((0,beads-1,0))
        self.setChains(chain)
        return np.vstack([x,y,z]).T
    
    def random_type(self, N): ##REMEMBER: 0,1,2,3,4,5 is A1,A2,B1,B2,B3,NA
        return random.choices(population=[0,1,2,3,4,5], k=N)
    
    def translate_type(self, filename):
        Type_conversion = {'A1':0, 'A2':1, 'B1':2, 'B2':3,'B3':4,'B4':5, 'NA' :6}
        my_list = []
        af = open(filename,'r')
        pos = af.read().splitlines()
        for t in range(len(pos)):
            pos[t] = pos[t].split()
            my_list.append(Type_conversion[pos[t][1]])
        
        self.type_list = my_list

    def create_line(self,N, sig=1.0):
        beads = N
        x = []
        y = []
        z = []
        for i in range(beads):
            x.append(0.15*sig*beads+(i-1)*0.6)
            y.append(0.15*sig*beads+(i-1)*0.6)
            z.append(0.15*sig*beads+(i-1)*0.6)
        
        chain = []
        chain.append((0,N-1,0))
        self.setChains(chain)

        return np.vstack([x,y,z]).T
    
##############################################################################################                    
      ############################### FUNCTIONS FOR SAVE ###############################
##############################################################################################

    def initStorage(self, filename, mode="w"):
        """
        filename : str
            Filename of an h5dict storage file

        mode :
            'w'  - Create file, truncate if exists
            'w-' - Create file, fail if exists         (default)
            'r+' - Continue simulation, file must exist.
        """
        import h5py
        
        self.storage = []

        if mode not in ['w', 'w-', 'r+']:
            raise ValueError("Wrong mode to open file."
                             " Only 'w','w-' and 'r+' are supported")
        if (mode == "w-") and os.path.exists(filename):
            raise IOError("Cannot create file... file already exists."                          " Use mode ='w' to override")
        for k, chain in zip(range(len(self.chains)),self.chains):
            fname = os.path.join(self.folder, filename + '_' +str(k) + '.cndb')
            self.storage.append(h5py.File(fname, mode))    
            self.storage[k]['types'] = self.type_list[chain[0]:chain[1]]
        #self.storage['loops'] = self.loopPosition

        if mode == "r+":
            myKeys = []
            for i in list(self.storage.keys()):
                try:
                    myKeys.append(int(i))
                except:
                    pass
            maxkey = max(myKeys) if myKeys else 1
            self.step = maxkey - 1
            self.setData(self.storage[str(maxkey - 1)])

                    
    def save(self, filename=None, mode="auto", h5dictKey="1", pdbGroups=None):
        data = self.getData()

        
        if filename is None:
            filename = self.name +"_block%d." % self.step + mode

        filename = os.path.join(self.folder, filename)
        
        if not hasattr(self, "type_list"): #if any list exist, create a random one!
             self.type_list = self.random_type(self.N)
        
        if mode == "auto":
            if hasattr(self, "storage"):
                mode = "h5dict"

        if mode == "h5dict":
            if not hasattr(self, "storage"):
                raise Exception("Cannot save to h5dict!"                                    " Initialize storage first!")
            for k, chain in zip(range(len(self.chains)),self.chains):
                self.storage[k][str(self.step)] = data[chain[0]:chain[1]]
            return
        
        elif mode == "txt":
            lines = []
            lines.append(str(len(data)) + "\n")

            for particle in data:
                lines.append("{0:.3f} {1:.3f} {2:.3f}\n".format(*particle))
            if filename == None:
                return lines
            elif isinstance(filename, six.string_types):
                with open(filename, 'w') as myfile:
                    myfile.writelines(lines)
            else:
                return lines

        elif mode == 'pdb':
            
            def add(st, n):
                if len(st) > n:
                    return st[:n]
                else:
                    return st + " " * (n - len(st) )
            
            for ncadeia, cadeia in zip(range(len(self.chains)),self.chains):
                
                if len(self.chains)>1: 
                    filename = self.name +"_" + str(ncadeia) + "_block%d." % self.step + mode
                    filename = os.path.join(self.folder, filename)
                else:
                    print('single chain')
                
                data_chain = data[cadeia[0]:cadeia[1]+1] 

                retret = ""
                          
                pdbGroups = ["A" for i in range(len(data_chain))]
                                                    
                for i, line, group in zip(list(range(len(data))), data_chain, pdbGroups):
                    atomNum = (i + 1) % 9000
                    segmentNum = (i + 1) // 9000 + 1
                    line = [float(j) for j in line]
                    ret = add("ATOM", 6)
                    ret = add(ret + "{:5d}".format(atomNum), 11)
                    ret = ret + " "
                    ret = add(ret + "CA", 17)
                    if (self.type_list[atomNum-1] == 0):
                        ret = add(ret + "ASP", 21)
                    elif (self.type_list[atomNum-1] == 1):
                        ret = add(ret + "GLU", 21)
                    elif (self.type_list[atomNum-1] == 2):
                        ret = add(ret + "HIS", 21)
                    elif (self.type_list[atomNum-1] == 3):
                        ret = add(ret + "LYS", 21)
                    elif (self.type_list[atomNum-1] == 4):
                        ret = add(ret + "ARG", 21)
                    elif (self.type_list[atomNum-1] == 5):
                        ret = add(ret + "ARG", 21)
                    elif (self.type_list[atomNum-1] == 6):
                        ret = add(ret + "ASN", 21)
                    ret = add(ret + group[0] + " ", 22)
                    ret = add(ret + str(atomNum), 26)
                    ret = add(ret + "        ", 30)
                    #ret = add(ret + "%i" % (atomNum), 30)
                    ret = add(ret + ("%8.3f" % line[0]), 38)
                    ret = add(ret + ("%8.3f" % line[1]), 46)
                    ret = add(ret + ("%8.3f" % line[2]), 54)
                    ret = add(ret + (" 1.00"), 61)
                    ret = add(ret + str(float(i % 8 > 4)), 67)
                    ret = add(ret, 73)
                    ret = add(ret + str(segmentNum), 77)
                    retret += (ret + "\n")
                with open(filename, 'w') as f:
                    f.write(retret)
                    #f.flush() 
                    
                    
        elif mode == 'gro':
            
            def add(st, n):
                if len(st) > n:
                    return st[:n]
                else:
                    return st + " " * (n - len(st) )
            
            for ncadeia, cadeia in zip(range(len(self.chains)),self.chains):
                filename = self.name +"_" + str(ncadeia) + "_block%d." % self.step + mode

                filename = os.path.join(self.folder, filename)
                data_chain = data[cadeia[0]:cadeia[1]+1] 

                retret = ""
                          
                pdbGroups = ["A" for i in range(len(data_chain))]
                                                    
                for i, line, group in zip(list(range(len(data))), data_chain, pdbGroups):
                    atomNum = (i + 1) % 9000
                    segmentNum = (i + 1) // 9000 + 1
                    line = [float(j) for j in line]
                    ret = add("ATOM", 6)
                    ret = add(ret + "{:5d}".format(atomNum), 11)
                    ret = ret + " "
                    ret = add(ret + "CA", 17)
                    if (self.type_list[atomNum-1] == 0):
                        ret = add(ret + "ASP", 21)
                    elif (self.type_list[atomNum-1] == 1):
                        ret = add(ret + "GLU", 21)
                    elif (self.type_list[atomNum-1] == 2):
                        ret = add(ret + "HIS", 21)
                    elif (self.type_list[atomNum-1] == 3):
                        ret = add(ret + "LYS", 21)
                    elif (self.type_list[atomNum-1] == 4):
                        ret = add(ret + "ARG", 21)
                    elif (self.type_list[atomNum-1] == 5):
                        ret = add(ret + "ARG", 21)
                    elif (self.type_list[atomNum-1] == 6):
                        ret = add(ret + "ASN", 21)
                    ret = add(ret + group[0] + " ", 22)
                    ret = add(ret + str(atomNum), 26)
                    ret = add(ret + "        ", 30)
                    #ret = add(ret + "%i" % (atomNum), 30)
                    ret = add(ret + ("%8.3f" % line[0]), 38)
                    ret = add(ret + ("%8.3f" % line[1]), 46)
                    ret = add(ret + ("%8.3f" % line[2]), 54)
                    ret = add(ret + (" 1.00"), 61)
                    ret = add(ret + str(float(i % 8 > 4)), 67)
                    ret = add(ret, 73)
                    ret = add(ret + str(segmentNum), 77)
                    retret += (ret + "\n")
                with open(filename, 'w') as f:
                    f.write(retret)
                    #f.flush() 

                
################################################################################
    ######### MINIMIZATION AND RUN FEATURES ################################
################################################################################

    def localEnergyMinimization(self, tolerance=0.3, maxIterations=0, random_offset=0.02):
        """        
        A wrapper to the build-in OpenMM Local Energy Minimization
        
        See caveat below 

        Parameters
        ----------
        
        tolerance: float 
            It is something like a value of force below which 
            the minimizer is trying to minimize energy to.             
            see openmm documentation for description 
            
            Value of 0.3 seems to be fine for most normal forces. 
            
        maxIterations: int
            Maximum # of iterations for minimization to do.
            default: 0 means there is no limit
            
            This is relevant especially if your simulation does not have a 
            well-defined energy minimum (e.g. you want to simulate a collapse of a chain 
            in some potential). In that case, if you don't limit energy minimization, 
            it will attempt to do a whole simulation for you. In that case, setting 
            a limit to the # of iterations will just stop energy minimization manually when 
            it reaches this # of iterations. 
            
        random_offset: float 
            A random offset to introduce after energy minimization. 
            Should ideally make your forces have realistic values. 
            
            For example, if your stiffest force is polymer bond force
            with "wiggle_dist" of 0.05, setting this to 0.02 will make
            separation between monomers realistic, and therefore will 
            make force values realistic. 
            
            See why do we need it in the caveat below. 
            
            
        Caveat
        ------
        
        If using variable langevin integrator after minimization, a big error may 
        happen in the first timestep. The reason is that enregy minimization 
        makes all the forces basically 0. Variable langevin integrator measures
        the forces and assumes that they are all small - so it makes the timestep 
        very large, and at the first timestep it overshoots completely and energy goes up a lot. 
        
        The workaround for now is to randomize positions after energy minimization 
        
        """

        print("Performing local energy minimization")

        self._applyForces()
        oldName = self.name
        self.name = "minim"

        self.state = self.context.getState(getPositions=False,
                                           getEnergy=True)
        eK = (self.state.getKineticEnergy() / self.N / units.kilojoule_per_mole)
        eP = self.state.getPotentialEnergy() / self.N/ units.kilojoule_per_mole
        locTime = self.state.getTime()
        print("before minimization eK={0}, eP={1} per monomers, time={2}".format(eK, eP, locTime))

        self.mm.LocalEnergyMinimizer.minimize(
            self.context, tolerance, maxIterations)

        self.state = self.context.getState(getPositions=True,
                                           getEnergy=True)
        eK = (self.state.getKineticEnergy() / self.N / units.kilojoule_per_mole)
        eP = self.state.getPotentialEnergy() / self.N / units.kilojoule_per_mole
        coords = self.state.getPositions(asNumpy=True)
        self.data = coords
        self.setData(self.getData(), random_offset = random_offset)        
        locTime = self.state.getTime()
        print("after minimization eK={0}, eP={1}, time={2}".format(eK, eP, locTime))

        self.name = oldName
    
    def energyMinimization(self, stepsPerIteration=100,
                           maxIterations=1000,
                           failNotConverged=True):
        """Runs system at smaller timestep and higher collision
        rate to resolve possible conflicts.

        this is here for backwards compatibility.
        """

        print("Performing energy minimization")
        self._applyForces()
        oldName = self.name
        self.name = "minim"
        if (maxIterations is True) or (maxIterations is False):
            raise ValueError(
                "Please stop using the old notation and read the new energy minimization code")
        if (failNotConverged is not True) and (failNotConverged is not False):
            raise ValueError(
                "Please stop using the old notation and read the new energy minimization code")

        def_step = self.integrator.getStepSize()
        def_fric = self.integrator.getFriction()

        def minimizeDrop():
            drop = 10.
            for dummy in range(maxIterations):
                if drop < 1:
                    drop = 1.
                if drop > 10000:
                    raise RuntimeError("Timestep too low. Perhaps, "                                       "something is wrong!")

                self.integrator.setStepSize(def_step / float(drop))
                self.integrator.setFriction(def_fric * drop)
                # self.reinitialize()
                numAttempts = 5
                for attempt in range(numAttempts):
                    a = self.doBlock(stepsPerIteration, increment=False,
                        reinitialize=False)
                    # self.initVelocities()
                    if a == False:
                        drop *= 2
                        print("Timestep decreased {0}".format(1. / drop))
                        self.initVelocities()
                        break
                    if attempt == numAttempts - 1:
                        if drop == 1.:
                            return 0
                        drop /= 2
                        print("Timestep decreased by {0}".format(drop))
                        self.initVelocities()
            return -1

        if failNotConverged and (minimizeDrop() == -1):
            raise RuntimeError(
                "Reached maximum number of iterations and still not converged\n"\
                "increase maxIterations or set failNotConverged=False")
        self.name = oldName
        self.integrator.setFriction(def_fric)
        self.integrator.setStepSize(def_step)
        # self.reinitialize()
        print("Finished energy minimization")
        
    def doBlock(self, steps=None, increment=True, num=None, reinitialize=True, maxIter=0, output=True, checkFunctions=[]):
        """performs one block of simulations, doing steps timesteps,
        or steps_per_block if not specified.

        Parameters
        ----------

        steps : int or None
            Number of timesteps to perform. If not specified, tries to
            infer it from self.steps_per_block
        increment : bool, optional
            If true, will not increment self.steps counter
        num : int or None, optional
            If specified, will split the block in num subblocks.
            Default value is 10.
        """

        if self.forcesApplied == False:
            if self.verbose:
                print("applying forces")
                stdout.flush()
            self._applyForces()
            self.forcesApplied = True
        if increment == True:
            self.step += 1
        if steps is None:
            steps = self.steps_per_block
        if (increment == True) and ((self.step % 50) == 0) and output:
            self.printStats()

        for attempt in range(15):
            if output:
                print("bl=%d" % (self.step), end=' ')
                stdout.flush()
            if self.verbose:
                print()
                stdout.flush()

            if num is None:
                num = steps // 5 + 1
            a = time.time()
            for _ in range(steps // num):
                if self.verbose:
                    print("performing integration")
                self.integrator.step(num)  # integrate!
                stdout.flush()
            if (steps % num) > 0:
                self.integrator.step(steps % num)

            # get state of a system: positions, energies
            self.state = self.context.getState(getPositions=True,
                                               getEnergy=True)

            b = time.time()
            coords = self.state.getPositions(asNumpy=True)
            newcoords = coords / nm

            # calculate energies in KT/particle
            eK = (self.state.getKineticEnergy() / self.N / units.kilojoule_per_mole)
            eP = self.state.getPotentialEnergy() / self.N / units.kilojoule_per_mole


            if self.velocityReinitialize:
                if eK > 5.4:
                    if output:
                        print("(i)", end=' ')
                    self.initVelocities()
            if output:
                print("pos[1]=[%.1lf %.1lf %.1lf]" % tuple(newcoords[0]), end=' ')



            checkFail = False
            for checkFunction in checkFunctions:
                if not checkFunction(newcoords):
                    checkFail = True

            if ((np.isnan(newcoords).any()) or (eK > self.eKcritical) or
                (np.isnan(eK)) or (np.isnan(eP))) or checkFail:
                self.printForces()
                self.save(mode = 'pdb', pdbGroups=self.chains)
                self.context.setPositions(self.data)
                self.initVelocities()
                if reinitialize == False:
                    if output:
                        print("eK={0}, eP={1}".format(eK, eP))
                    return False
                if output:
                    print("eK={0}, eP={1}, trying one more time at step {2} ".format(eK, eP, self.step))
            else:
                dif = np.sqrt(np.mean(np.sum((newcoords -
                    self.getData()) ** 2, axis=1)))
                self.data = coords
                if output:
                    print("dr=%.2lf" % (dif,), end=' ')
                    print("t=%2.1lfps" % (self.state.getTime() / ps), end=' ')
                    print("kin=%.2lf pot=%.2lf" % (eK,
                    eP), "Rg=%.3lf" % self.RG(), end=' ')
                    print("SPS=%.0lf" % (steps / (float(b - a))), end=' ')

                if (self.integrator_type.lower() == 'variablelangevin'
                    or self.integrator_type.lower() == 'variableverlet'):
                    dt = self.integrator.getStepSize()
                    print('dt=%.1lffs' % (dt / fs), end=' ')
                    mass = self.system.getParticleMass(1)
                    dx = (units.sqrt(2.0 * eK * self.kT / mass) * dt)
                    print('dx=%.2lfpm' % (dx / nm * 1000.0), end=' ')

                if output:
                    print("")
                break

            if attempt in [3, 4, 5, 6, 7 , 8, 9, 10 ,11 ,12 ,13 ,14]:
                self.localEnergyMinimization(maxIterations=maxIter)
            if attempt == 15:
                self._exitProgram("exceeded number of attempts")

        return {"Ep":eP, "Ek":eK}
        
        
    def initPositions(self):
        """Sends particle coordinates to OpenMM system.
        If system has exploded, this is
         used in the code to reset coordinates. """

        print("Positions... ")
        try:
            self.context
        except:
            raise ValueError("No context, cannot set velocs."                             " Initialize context before that")

        self.context.setPositions(self.data)
        print(" loaded!")
        state = self.context.getState(getPositions=True, getEnergy=True)
            # get state of a system: positions, energies
        eP = state.getPotentialEnergy() / self.N / units.kilojoule_per_mole
        print("potential energy is %lf" % eP)
        
    def initVelocities(self, mult=1):
        """Initializes particles velocities

        Parameters
        ----------
        mult : float, optional
            Multiply velosities by this. Is good for a cold/hot start.
        """
        try:
            self.context
        except:
            raise ValueError("No context, cannot set velocs."                             "Initialize context before that")

        sigma = units.sqrt(self.Epsilon*units.kilojoule_per_mole / self.system.getParticleMass(
            1))  # calculating mean velocity
        velocs = units.Quantity(mult * np.random.normal(
            size=(self.N, 3)), units.meter) * (sigma / units.meter)
        # Guide to simtk.unit: 1. Always use units.quantity.
        # 2. Avoid dimensionless shit.
        # 3. If you have to, create fake units, as done here with meters
        self.context.setVelocities(velocs)
        
    def setfibposition(self, mypol, plot=True, dist=(1.0,3.0)):
        def fibonacci_sphere(samples=1, randomize=True):
            import math
            rnd = 1.
            if randomize:
                rnd = random.random() * samples

            points = []
            offset = 2./samples
            increment = math.pi * (3. - math.sqrt(5.))

            for i in range(samples):
                y = ((i * offset) - 1) + (offset / 2)
                r = math.sqrt(1 - pow(y,2))
                phi = ((i + rnd) % samples) * increment
                x = math.cos(phi) * r
                z = math.sin(phi) * r
                points.append([x,y,z])
            return points
    
        def plotdistrivuition(points, filename='.'):
            from mpl_toolkits.mplot3d import Axes3D
            r = 1
            pi = np.pi
            cos = np.cos
            sin = np.sin
            phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
            x = r*sin(phi)*cos(theta)
            y = r*sin(phi)*sin(theta)
            z = r*cos(phi)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            xx=np.array(points)[:,0]
            yy=np.array(points)[:,1]
            zz=np.array(points)[:,2]
            ax.plot_surface(
                x, y, z,  rstride=1, cstride=1, color='c', alpha=0.4, linewidth=0)

            ax.scatter(xx,yy,zz,color="red",s=80)
            plt.savefig(filename + '.png')
    
        points = fibonacci_sphere(len(self.chains))
        R_nucleus = ( (self.chains[-1][1]+1) * (1.0/2.)**3 / 0.1 )**(1./3)
        if (plot):
            filename = "chainsDistr_%d." % self.step
            filename = os.path.join(self.folder, filename)
            plotdistrivuition(points=points, filename=filename)
        
        for i in range(len(self.chains)):
            points[i] = [ x * dist[0] * R_nucleus + dist[1] * R_nucleus for x in points[i]]
            mypol[self.chains[i][0]:self.chains[i][1]+1] -= np.array(points[i])
            
        return(mypol)
        
    def RG(self):
        """
        Returns
        -------

        Gyration ratius in units of length (bondlength).
        """
        data = self.getScaledData()
        data = data - np.mean(data, axis=0)[None,:]
        return np.sqrt(np.sum(np.var(np.array(data), 0)))
    
    def getScaledData(self):
        """Returns data, scaled back to PBC box """
        if self.PBC != True:
            return self.getData()
        alldata = self.getData()
        boxsize = numpy.array(self.BoxSizeReal)
        mults = numpy.floor(alldata / boxsize[None, :])
        toRet = alldata - mults * boxsize[None, :]
        assert toRet.min() >= 0
        return toRet
    
    def printStats(self):
        """Prints detailed statistics of a system.
        Will be run every 50 steps
        """
        state = self.context.getState(getPositions=True,
            getVelocities=True, getEnergy=True)

        eP = state.getPotentialEnergy()
        pos = np.array(state.getPositions() / nm)
        bonds = np.sqrt(np.sum(np.diff(pos, axis=0) ** 2, axis=1))
        sbonds = np.sort(bonds)
        vel = state.getVelocities()
        mass = self.system.getParticleMass(1)
        vkT = np.array(vel / units.sqrt(self.Epsilon*units.kilojoule_per_mole / mass), dtype=float)
        self.velocs = vkT
        EkPerParticle = 0.5 * np.sum(vkT ** 2, axis=1)

        cm = np.mean(pos, axis=0)
        centredPos = pos - cm[None, :]
        dists = np.sqrt(np.sum(centredPos ** 2, axis=1))
        per95 = np.percentile(dists, 95)
        den = (0.95 * self.N) / ((4. * np.pi * per95 ** 3) / 3)
        per5 = np.percentile(dists, 5)
        den5 = (0.05 * self.N) / ((4. * np.pi * per5 ** 3) / 3)
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        minmedmax = lambda x: (x.min(), np.median(x), x.mean(), x.max())

        print()
        print("Statistics for the simulation %s, number of particles: %d, "        " number of chains: %d" % (
            self.name, self.N, len(self.chains)))
        print()
        print("Statistics for particle position")
        print("     mean position is: ", np.mean(
            pos, axis=0), "  Rg = ", self.RG())
        print("     median bond size is ", np.median(bonds))
        print("     three shortest/longest (<10)/ bonds are ", sbonds[
            :3], "  ", sbonds[sbonds < 10][-3:])
        if (sbonds > 10).sum() > 0:
            print("longest 10 bonds are", sbonds[-10:])

        print("     95 percentile of distance to center is:   ", per95)
        print("     density of closest 95% monomers is:   ", den)
        print("     density of the core monomers is:   ", den5)
        print("     min/median/mean/max coordinates are: ")
        print("     x: %.2lf, %.2lf, %.2lf, %.2lf" % minmedmax(x))
        print("     y: %.2lf, %.2lf, %.2lf, %.2lf" % minmedmax(y))
        print("     z: %.2lf, %.2lf, %.2lf, %.2lf" % minmedmax(z))
        print()
        print("Statistics for velocities:")
        print("     mean kinetic energy is: ", np.mean(
            EkPerParticle), "should be:", 1.5)
        print("     fastest particles are (in kT): ", np.sort(
            EkPerParticle)[-5:])

        print()
        print("Statistics for the system:")
        print("     Forces are: ", list(self.forceDict.keys()))
        print("     Number of exceptions:  ", len(self.bondsForException))
        print()
        print("Potential Energy Ep = ", eP / self.N / units.kilojoule_per_mole)
        
    def printForces(self):
        from pandas import DataFrame

        forceNames = []
        forceValues = []
        
        for i,n in enumerate(self.forceDict):
            forceNames.append(n)
            forceValues.append(self.context.getState(getEnergy=True, groups={i}).getPotentialEnergy().value_in_unit(units.kilojoules_per_mole))
        forceNames.append('Potential Energy (total)')
        forceValues.append(self.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(units.kilojoules_per_mole))
        forceNames.append('Potential Energy (per mer)')
        forceValues.append(self.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(units.kilojoules_per_mole)/self.N)
        df = DataFrame(forceValues,forceNames)
        df.columns = ['Values(Kj/mol)']
        print(df)
        #print('\t'.join([str(e) for e in forceNames]))
        #print('\t'.join([str(e) for e in forceValues]))
        
    def getForces(self):
        from pandas import DataFrame

        forceNames = []
        forceValues = []
        
        for i,n in enumerate(self.forceDict):
            forceNames.append(n)
            forceValues.append(self.context.getState(getEnergy=True, groups={i}).getPotentialEnergy().value_in_unit(units.kilojoules_per_mole))
        forceNames.append('Potential Energy (total)')
        forceValues.append(self.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(units.kilojoules_per_mole))
        forceNames.append('Potential Energy (per mer)')
        forceValues.append(self.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(units.kilojoules_per_mole)/self.N)
        #df = DataFrame(forceValues,forceNames)
        #df.columns = ['Values(Kj/mol)']
        return forceNames, forceValues


class SerraAngel:

    ##initialize the Phage set
    '''
    This classe is used to optimize constants values for Open-Michron Potentials
    '''
 
    def __init__(self, state, TypeList=None, name='distMatrix', nHood=3): 
        self.name = name
        self.size = len(state)
        self.P=np.zeros((self.size,self.size))
        self.Pold=np.zeros((self.size,self.size))
        self.r_cut = 1.61 #1.78 #1.985907 #1.558225 #1.78       #parameter for contact function f
        self.mu  = 5.33  #3.22 #2.12096 #3.77805 #3.22   #parameter for contact function f
        self.Bij = np.zeros((200,200))
        self.polds_type = np.zeros(3)
        self.Bij_type = np.zeros((3,3))
        self.Nframes = 0 
        self.dinit = nHood
        
        
##########################################################################################
#### IDEAL CROMOSSOME OPTIMIZATION
##########################################################################################
    
    def probCalculation(self, state, dmax=200):
        #remember dinit = 3, i.e, Bij[0,1] = 3,4
        PiPj = np.zeros((200,200))
        self.Pold += self.P
        self.P = 0.5*(1.0 + np.tanh(self.mu*(self.r_cut - distance.cdist(state,state, 'euclidean'))))
        dmaxl = range(dmax)
        for i, j in itertools.product(dmaxl,dmaxl):
            PiPj[i,j] = np.mean(np.diagonal(self.P, offset=(i+self.dinit)))*np.mean(np.diagonal(self.P, offset=(j+self.dinit)))
      
        self.Bij += PiPj
        self.Nframes += 1 
        
    
    def calc_sim_phi(self, init=3, dmax=200):
        phi = np.zeros(200)
        pmean = self.Pold/self.Nframes
        for i in range(dmax):
             phi[i] =  np.mean(np.diagonal(pmean, offset=(i+init)))
        return phi
    
    def getBijsim(self):
        return self.Bij/self.Nframes
    
    
    def getHiCexp(self, filename):
        allmap = np.loadtxt(filename)

        from sklearn.preprocessing import normalize

        r=np.triu(allmap, k=1) #tirando a diagonal principal e pegando só a matriz superior
        r = normalize(r, axis=1, norm='max') #normalizando em função do maior valor
        rd = np.transpose(r) #criando a matriz triagular inferior
        self.expHiC = r+rd + np.diag(np.ones(len(r))) #somando tudo e adicionado 1 na diagonal princial
        self.expHiC[self.expHiC<0.3] = 0.0

    def calc_exp_phi(self, init=3, dmax=200):
        phi = np.zeros(200)
        for i in range(dmax):
             phi[i] =  np.mean(np.diagonal(self.expHiC, offset=(i+init)))
        return phi
    
    
    def getlambfromfile(self, filename):
        aFile = open(filename,'r')
        pos = aFile.read().splitlines()
        for t in range(len(pos)):
            pos[t] = float(pos[t])
        return np.array(pos)
    
    def getLamb(self,exp_map='/home/antonio/Felidae/puma/c03/puma_c03.dense'):
        
   
        self.getHiCexp(exp_map)
        #b.expHiC[b.expHiC<0.001] = 0.0 #COLOCAR ESSA LINhA
        
        phi_exp = self.calc_exp_phi()
        
        phi_sim = self.calc_sim_phi()
        
        gij = -phi_sim + phi_exp   # *1/beta = 1     
    
        Res = np.zeros((200,200))
        Bijmean = self.getBijsim()
       
        for i, j in itertools.product(range(200),range(200)):
            Res[i,j] = Bijmean[i,j] - (phi_sim[i]*phi_sim[j])
         
        invRes = sp.linalg.pinv(Res)
        
        print("erro: ", np.sum(np.absolute(gij))/np.sum(phi_exp))
        print("first inv", invRes[0,:5])
        print("first Res", Res[0,:5])
        print("first gij", gij[:5])
        
        
        return(np.dot(invRes,gij))
    
##########################################################################################
#### TYPES OPTIMIZATION
##########################################################################################


    def probCalculation_types(self, state, typeList=None):
        
        PiPj = np.zeros((3,3))
        p_actual = np.zeros(3)

        
        just_a = [i for i, e in enumerate(typeList) if e == 0]
        just_b = [i for i, e in enumerate(typeList) if e == 2]
        
        self.Pold += self.P       
        
        self.P = 0.5*(1.0 + np.tanh(self.mu*(self.r_cut - distance.cdist(state,state, 'euclidean'))))
        self.P[self.P<0.3] = 0.0
        
        nt=0
        for i, j in itertools.product(just_a,just_a):
            if j>i+1:
                p_actual[0] += self.P[i,j]
                nt += 1
        p_actual[0] = p_actual[0]/nt
        
        nt=0
        for i, j in itertools.product(just_a,just_b):
            if j>i+1:
                p_actual[1] += self.P[i,j]
                nt += 1
        p_actual[1] = p_actual[1]/nt
        
        nt=0
        for i, j in itertools.product(just_b,just_b):
            if j>i+1:
                p_actual[2] += self.P[i,j]
                nt += 1
        p_actual[2] = p_actual[2]/nt

        #p_actual[0] = np.mean(0.5*(1.0 + np.tanh(self.mu*(self.r_cut - distance.cdist(state[just_a],state[just_a], 'euclidean')))))
        #p_actual[1] = np.mean(0.5*(1.0 + np.tanh(self.mu*(self.r_cut - distance.cdist(state[just_a],state[just_b], 'euclidean')))))
        #p_actual[2] = np.mean(0.5*(1.0 + np.tanh(self.mu*(self.r_cut - distance.cdist(state[just_b],state[just_b], 'euclidean')))))
        #p_actual[3] = np.mean(0.5*(1.0 + np.tanh(self.mu*(self.r_cut - distance.cdist(state[just_b],state[just_b], 'euclidean')))))
        
        #print(p_actual[0],p_actual[1],p_actual[2])
        
        for i, j in itertools.product(range(3),range(3)):
            PiPj[i,j] = p_actual[i]*p_actual[j] 
        
        #print(PiPj)
        self.polds_type += p_actual 
        self.Bij_type += PiPj
        self.Nframes += 1
    
    def calc_exp_phi_types(self, typeList):
        phi = np.zeros(3)
        just_a = [i for i, e in enumerate(typeList) if e == 0]
        just_b = [i for i, e in enumerate(typeList) if e == 2]
        
        nt = 0
        for i, j in itertools.product(just_a,just_a):
            if j>i+1:
                phi[0] += self.expHiC[i,j]
                nt += 1
        phi[0] = phi[0]/nt
        
        nt = 0
        for i, j in itertools.product(just_a,just_b):
            if j>i+1:
                phi[1] += self.expHiC[i,j]
                nt += 1
        phi[1] = phi[1]/nt
        
        nt = 0
        for i, j in itertools.product(just_b,just_b):
            if j>i+1:
                phi[2] += self.expHiC[i,j]
                nt += 1
        phi[2] = phi[2]/nt
        
        
        return phi
    
    
    def calc_sim_phi_types(self):
        return self.polds_type/self.Nframes
    
    
    def getBijsim_types(self):
        return self.Bij_type/self.Nframes
    
    def getHiCSim(self):
        return self.Pold/self.Nframes
    
    def getPearson(self):
        r1 = self.getHiCSim()
        r2 = self.expHiC

        r1[np.isinf(r1)]= 0.0
        r1[np.isnan(r1)]= 0.0
        r1[r1 <= 0.001]= 0.0
        r2[np.isinf(r2)]= 0.0
        r2[np.isnan(r2)]= 0.0
        r2[r2<=0.001] = 0.0

        np.fill_diagonal(r1,0.0)
        np.fill_diagonal(r2,0.0)


        SEED = 100
        random.seed(SEED)
        a1 = np.asarray(random.sample(list(r1[np.triu_indices(np.shape(r1)[0])]),np.int(0.1*np.shape(r1)[0]*np.shape(r1)[0])))
        a1 = r1[np.triu_indices(np.shape(r1)[0])]
        random.seed(SEED)
        a2 = np.asarray(random.sample(list(r2[np.triu_indices(np.shape(r2)[0])]),np.int(0.1*np.shape(r2)[0]*np.shape(r2)[0])))
        a2 = r2[np.triu_indices(np.shape(r1)[0])]

        from scipy.stats.stats import pearsonr
        return(pearsonr(a1,a2)[0])
        
    def getLamb_types(self,typeList, exp_map='puma_c03.dense'):
        
        self.getHiCexp(exp_map) #ok
        
        phi_exp = self.calc_exp_phi_types(typeList)  #ok
        
        phi_sim = self.calc_sim_phi_types() #ok
        
        gij = -phi_sim + phi_exp   # *1/beta = 1    #ok 
    
        Res = np.zeros((3,3))
        Bijmean = self.getBijsim_types()
       
        for i, j in itertools.product(range(3),range(3)):
            Res[i,j] = Bijmean[i,j] - (phi_sim[i]*phi_sim[j])
         
        invRes = sp.linalg.pinv(Res)
        
        erro = np.sum(np.absolute(gij))/np.sum(phi_exp)
        pear = self.getPearson()
        
                             
        with open('/home/antonio/Desktop/erro','a') as tf:
            tf.write("Erro: %f  Pearson Correlation: %f\n" % (erro, pear))
                
                
        #print("first inv", invRes[0,:5])
        #print("first Res", Res[0,:5])
        #print("first gij", gij[:5])
        
        
        return(np.dot(invRes,gij))        
                                                                                                
