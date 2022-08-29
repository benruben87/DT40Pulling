from openmichrolib import Akroma as openmichrom
from openmichrolib import SerraAngel as trainertool
import time
import numpy as np
import time
import h5py
from   scipy.spatial  import distance
import sys
from numpy.random import random as rand
from numpy.linalg import norm
from itertools import islice, product, combinations
import matplotlib.pyplot as plt
import os
from scipy.ndimage.filters import uniform_filter1d
import matplotlib as mpl

def calcContProbs(fileName, mu, r_cut, eqFrames, chronum = 7):
    
    start         = time.time()
    chro = h5py.File(fileName, 'r') # comando para abrir o arquivo
    frames = len(chro.keys()) - 1
    firstkey = list(chro.keys())[0]
    
    keylist = list(chro.keys())
    flabels = np.sort([int(key) for key in keylist[0:len(keylist)-1]])
    
    P = np.zeros((len(chro[firstkey]), len(chro[firstkey])))

    print('####################################################################')
    print('Making Contact Map of Chromosome: Chro {:} with {:} beads'.format(chronum, (len(chro[firstkey]))))
    
    ctr=0
    for i,label in enumerate(flabels):
        XYZ = np.asarray(chro[str(label)])

        if i >eqFrames:
            D = distance.cdist(XYZ, XYZ, 'euclidean')
            D[D<=1.0] = 0.0
            P += 0.5 * (1.0 + np.tanh(mu * (r_cut - D)))
            ctr+=1

        end     = time.time()
        elapsed = end - start
        #print('Frame {:} : %.3f s'.format(i) % elapsed)

        if i % 500 == 0:
            print("Reading frame", i)

    print(i)
    
    CMap = np.divide(P , ctr)
    
    chro.close()
    
    end     = time.time()
    elapsed = end - start
    
    print('Total frames: {:}'.format(i))
    print('############################################################')
    
    return CMap


#Given a frame of coordinates and a renormalization scale n (number of beads per group), returns the mean coarse-grained bond length of the chain, Bond Cosines (mean and std), and the average radius of gyration of the coarse-grained beads of size n.
def calc_lens_cos_rg(pos,n):

    org = -int(n/2)
    
    #Super Fast Renormalization with scipy
    if n>1:
        renrm = uniform_filter1d(pos, size = n, axis = 0, origin = -int(n/2))[0:-n+1]
    else:
        renrm = pos
    
    bonds = np.subtract(renrm[n:], renrm[:-n])
    bondLens = np.sqrt(np.matmul(np.multiply(bonds, bonds), np.ones(3).reshape((-1,1))))
    meanBondLength = np.mean(bondLens)
    unitTangents = np.divide(bonds, np.hstack([bondLens, bondLens, bondLens]))
    
    
    
    #Cosine Matrix is the sum of outer products of the x y and z components of the unit tangents
    CosineMatrix = np.outer(unitTangents[:,0],unitTangents[:,0]) + np.outer(unitTangents[:,1],unitTangents[:,1]) + np.outer(unitTangents[:,2], unitTangents[:,2])
    
    ContourLens = 1/n*np.multiply(meanBondLength,list(range(len(CosineMatrix))))
    
    cos_mean,cos_std=[],[]
    
    for diff in range(len(CosineMatrix)):
        cosines = np.diagonal(CosineMatrix,diff)
        cos_mean.append(np.mean(cosines))
        cos_std.append(np.std(cosines))

    ##Calculate rg of coarse-grained beads:
    
    if n==1:
        allrgs = [0 for _ in range(len(renrm))]
    else:
        allrgs = []
        diffs = pos[:n]-renrm[0]
        Rg = np.sqrt(np.mean(np.matmul(np.multiply(diffs, diffs), np.ones((3,1)))))
        
        allrgs.append(Rg)
        
        for x in range(len(renrm)-1):
            Rg = np.sqrt(Rg**2 - 1/n**2*np.dot(pos[x+n]-pos[x], pos[x+n]-pos[x]) + 1/n*(np.dot(pos[x+n]-renrm[x], pos[x+n]-renrm[x])-np.dot(pos[x]-renrm[x],pos[x]-renrm[x])))
            #diffs = pos[x:n+x]-renrm[x]
            #Rg = np.sqrt(np.mean(np.matmul(np.multiply(diffs, diffs), np.ones((3,1)))))
            
            allrgs.append(Rg)
        
    rg_mean= np.mean(allrgs)
    rg_std = np.std(allrgs)
    
    return ContourLens,cos_mean,cos_std, rg_mean, rg_std

#Calculates renormalized bond lengths, bond cosines, coarse-grained bead rg for a whole trajectory.  Splits the trajectory into numGroups sections to check for equilibration of these quantities.
def calc_lens_cos_rg_split_traj(trajFile, n_arr = [1, 5, 20, 50, 100, 200], skip_arr = [], numGroups = 1, eqFrames = 0):
    
    print('############### Calculating Len Cos Rg ###################')
    
    #If Skips are not specified, assigns them all to 1.  
    #Skips are the number of frames to skip between calculating cosines.
    if len(skip_arr)==0:
        skip_arr = [1 for _ in n_arr]
    
    start=time.time()
    
    chro = 7
    chro1 = h5py.File(trajFile, 'r') # comando para abrir o arquivo

    frames = len(chro1.keys()) - 1
    firstkey = list(chro1.keys())[0]
    
    print('Chromosome: Chro {:} with {:} beads'.format(chro, (len(chro1[firstkey]))))

    keylist = list(chro1.keys())
    #print(keylist)
    AllFLabels = np.sort([int(key) for key in keylist[0:len(keylist)-1]])
    #print(AllFLabels)
    
    FramesPerGroup = (frames-eqFrames) // numGroups
    
    Time_Group_Dict = {}
    Time_Group_Dict['FPG'] = FramesPerGroup
    
    ticker = eqFrames # index of current label in AllFLabels.  Starts at eqFrames
    for curGroup in range(numGroups):
    
        Len_Cos_RG_Dict = {}
        ctr = np.zeros(len(n_arr))#Counts the number of samples for each n

        for p in range(FramesPerGroup):

            flabel = AllFLabels[ticker]

            if (ticker+1) % 500 == 0:
                print('Analyzing Frame: ' + str(ticker+1) + ' out of ' + str(frames))

            ticker+=1

            XYZ = np.asarray(chro1[str(flabel)])

            for j,n in enumerate(n_arr):
                if np.mod(ticker,skip_arr[j])==0:
                    ctr[j]+=1

                    ContourLens, cos_mean, cos_std, rg_mean, rg_std = calc_lens_cos_rg(XYZ,n)
                    if  not (n in Len_Cos_RG_Dict.keys()):
                        Len_Cos_RG_Dict[n] = {}
                        Len_Cos_RG_Dict[n]['len'] = ContourLens
                        Len_Cos_RG_Dict[n]['cos'] = cos_mean
                        Len_Cos_RG_Dict[n]['rg'] = rg_mean
                    else:
                        Len_Cos_RG_Dict[n]['len'] = np.add(Len_Cos_RG_Dict[n]['len'],ContourLens) 
                        Len_Cos_RG_Dict[n]['cos'] = np.add(Len_Cos_RG_Dict[n]['cos'],cos_mean)
                        Len_Cos_RG_Dict[n]['rg'] = np.add(Len_Cos_RG_Dict[n]['rg'], rg_mean)

        for j,n in enumerate(n_arr):
            Len_Cos_RG_Dict[n]['len'] = np.divide(Len_Cos_RG_Dict[n]['len'], ctr[j]).tolist()
            Len_Cos_RG_Dict[n]['cos'] = np.divide(Len_Cos_RG_Dict[n]['cos'],ctr[j]).tolist()
            Len_Cos_RG_Dict[n]['rg'] = np.divide(Len_Cos_RG_Dict[n]['rg'],ctr[j]).tolist()

        Time_Group_Dict[curGroup] = Len_Cos_RG_Dict

    end     = time.time()
    elapsed = end - start

    chro1.close()

    end     = time.time()
    elapsed = end - start

    print('Total frames: {:}'.format(ticker))
    print('Ran in %.3f sec' % elapsed)
    print('############################################################')
    return Time_Group_Dict

#Calculates renormalized bond lengths, bond cosines, coarse-grained bead rg for a whole trajectory.
def calc_lens_cos_rg_traj(trajFile, n_arr = [1, 5, 20, 50, 100, 200], skip_arr = [], eqFrames = 0):
    
    TimeGroupDict = calc_lens_cos_rg_split_traj(trajFile, n_arr, skip_arr, numGroups = 1, eqFrames = eqFrames)
    return TimeGroupDict[0]

#Takes mean of all values in len_cos_rg_dict's corresponding to the time groups given in the groups array.
def combineTimeGroupDicts(TimeGroupDict, groups = -1):
    if groups == -1:
        groups = list(Time_Group_Dict.keys())
        
    combDict = {}
    
    for curKey in Time_Group_Dict[groups[0]].keys():
        ctr = 0
        meanVal = 0
        for groupKey in groups:
            meanVal+=TimeGroupDict[groupKey][curKey]
            ctr+=1
        
        meanVal = meanVal/ctr
        
        combDict[curKey] = meanVal
    
    return combDict

#Calculates Renormalized BondLength as a function of renormalization scale b and returns mean bond lengths.  Bond length should increase with b, as cg beads get bigger.  However, they will in general increase slower than linearly.
#trajFile is cndb trajectory
#skipnm is the jump between frames.  Used to speed up calculation if necessary.
def calc_renormalized_bondlens_traj(trajFile,eqFrames = 0, skipnum=1, chro=7):
    
    print('############### Calculating Renormalized Bond Lengths ###################')
    start=time.time()
    
    chro1 = h5py.File(trajFile, 'r') # comando para abrir o arquivo

    frames = len(chro1.keys()) - 1
    firstkey = list(chro1.keys())[0]
    
    print('Chromosome: Chro {:} with {:} beads'.format(chro, (len(chro1[firstkey]))))

    keylist = list(chro1.keys())
    flabels = np.sort([int(key) for key in keylist[0:len(keylist)-1]])

    BondLens = []# Vector of mean bond lengths.  Index is the degree of coarse-graining
    
    for n in range(1, int(len(chro1[firstkey])//2+1)):
        
        if n%100 == 0:
            print('Renormalizing By ' + str(n))
        
        curMeanBondLens = []
        for i, flabel in enumerate(flabels):
            
            if i>eqFrames and np.mod(i, skipnum)==0:
        
                XYZ = np.asarray(chro1[str(flabel)])
            
                if n>1:
                    renrm = uniform_filter1d(XYZ, size = n, axis = 0, origin = -int(n/2))[0:-n+1]
                
                else:
                    renrm = XYZ
                    
                bonds = np.subtract(renrm[n:], renrm[:-n])
                meanBondLength = np.mean(np.sqrt(np.matmul(np.multiply(bonds, bonds), np.ones(3).reshape((-1,1)))))
                curMeanBondLens.append(meanBondLength)
                
        BondLens.append(np.mean(curMeanBondLens))
        
    end     = time.time()
    elapsed = end - start

    chro1.close()
    print('Total frames: {:}'.format(i))
    print('Ran in %.3f sec' % elapsed)
    print('############################################################')
    
    return BondLens

#Calculating Radial Distributions of Beads Relative to renormalized chain.

#Given a renormalized structure and full coordinates, returns a list of minimum distances between the coordinates and the renormalized chain.
#For speed, checks against the renormalized chain in a window of width w from the window on which the bead is centered.
#Ignores coordinates in the first b/2 or last b/2 of the full chain.
def getMinDists(renormalized, coords, w=500):
    MinDists = []
    b=len(coords)-len(renormalized)+1
    coords = coords[int(b/2):-int(b/2)]
    for i in range(len(coords)):
        lowerIndex = int(np.max([0, i-w]))
        upperIndex = int(np.min([len(renormalized)-1, i+w]))
        diffs = coords[i]-renormalized[lowerIndex:upperIndex]
        dists = np.sqrt(np.matmul(np.multiply(diffs, diffs),np.ones(3).reshape((-1,1))))
        MinDists.append(np.min(dists))
        
    return MinDists

def calc_relative_radial_dist_traj(trajFile, eqFrames = 0, skipnum=1, n=300):
    
    BinWidth = .01
    Edges = np.arange(0, 30+BinWidth, BinWidth)
    BinVals = np.zeros(len(Edges)-1)
    
    start=time.time()
    
    chro = 7
    chro1 = h5py.File(trajFile, 'r') # comando para abrir o arquivo

    frames = len(chro1.keys()) - 1
    firstkey = list(chro1.keys())[0]
    
    print(frames)
    print('Chromosome: Chro {:} with {:} beads'.format(chro, (len(chro1[firstkey]))))
    print('Window For Core: ' + str(n) + ' beads')
    keylist = list(chro1.keys())
    flabels = np.sort([int(key) for key in keylist[0:len(keylist)-1]])
    
    for i, flabel in enumerate(flabels):
            
        if i>eqFrames and np.mod(i, skipnum)==0:

            XYZ = np.asarray(chro1[str(flabel)])

            if n>1:
                renrm = uniform_filter1d(XYZ, size = n, axis = 0, origin = -int(n/2))[0:-n+1]

            else:
                renrm = XYZ

            curRadialDists = getMinDists(renrm, XYZ, w=n)
            curBinVals,_ = np.histogram(curRadialDists, Edges)
            BinVals = np.add(BinVals, curBinVals)
    
    #Normalize Bin Values
    BinVals = np.divide(BinVals, BinWidth*np.sum(BinVals))
    histData=[Edges.tolist(), BinVals.tolist()]
    end = time.time()
    
    elapsed = end-start
    
    chro1.close()
    print('Total frames: {:}'.format(i))
    print('Ran in %.3f sec' % elapsed)
    print('############################################################')
    
    return histData


#Calculating the radial distribution of beads in the space of the chosen axes.

#Given full coordinates, returns a list of distances between the center of mass and individual beads projected into the line, plane, or 3D space chosen by the input dimensions.
#axes = [1,1,1] is appropriate for the radius of a sphere (interphase dry runs)
#axes = [0,1,1] is appropriate for the radius of a cylinder oriented along the x-axis.
def getRadialDists(coords, axes = [1,1,1]):
    Dists = []
    COM = np.mean(coords, axis = 0)

    for i in range(len(coords)):
        
        diff = coords[i]-COM
        Dists.append(np.sqrt(np.matmul(np.multiply(diff, diff),np.array(axes).reshape((-1,1)))))
        
    return Dists


def calc_axis_radial_dist_traj(trajFile, skipnum=50, axes = [1,1,1], eqFrames = 0, chro=7):
    
    start=time.time()
    
    chro1 = h5py.File(trajFile, 'r') # comando para abrir o arquivo

    frames = len(chro1.keys()) - 1
    firstkey = list(chro1.keys())[0]
    
    print(frames)
    print('Chromosome: Chro {:} with {:} beads'.format(chro, (len(chro1[firstkey]))))

    keylist = list(chro1.keys())
    flabels = np.sort([int(key) for key in keylist[0:len(keylist)-1]])
    
    BinWidth = .01
    Edges = np.arange(0, 30+BinWidth, BinWidth)
    BinVals = np.zeros(len(Edges)-1)
    
    for i, flabel in enumerate(flabels):
            
        if i>eqFrames and np.mod(i, skipnum)==0:

            XYZ = np.asarray(chro1[str(flabel)])

            curRadialDists = getRadialDists(XYZ, axes)
            curBinVals,_ = np.histogram(curRadialDists, Edges)
            BinVals = np.add(BinVals, curBinVals)

    #Normalize Bin Values
    BinVals = np.divide(BinVals, BinWidth*np.sum(BinVals))
    histData=[Edges.tolist(), BinVals.tolist()]
    
    chro1.close()
    end = time.time()
    elapsed = end-start
    print('Total frames: {:}'.format(i))
    print('Ran in %.3f sec' % elapsed)
    print('############################################################')
    
    return histData

def find_opt_bead_size(trajFile, BondLens = [], eqFrames = 0, N_init = 600, volFraction = .95):
    
    if len(BondLens) == 0:
        BondLens = calc_renormalized_bondlens_traj(trajFile, skipnum=1)
    
    N_old = -1 #initialize to something non-zero
    N = N_init # Initial Guess for the coarse-grained bead size.

    print('##############################')
    print('Initial Guess for N: ' + str(N_init))
    ctr = 0
    
    while N != N_old:

        radHistData = calc_relative_radial_dist_traj(trajFile, eqFrames = eqFrames, n=N)
        Bins = radHistData[0]
        Centers = .5*np.add(Bins[1:],Bins[0:len(Bins)-1])

        CumBinVals = np.cumsum(radHistData[1])*.01
        radius = Centers[np.argmin(np.abs(CumBinVals-volFraction))]
        N_old = N
        N = np.argmin(np.abs(BondLens-2*radius))+1 # plus one because indices start at 0 but cg bead sizes starts at 1
        rad_cyl = radius
        rad_bead = .5*BondLens[N-1]
        
        ctr+=1
        print('### After ' + str(ctr) + ' Iterations:')
        print('N = ' + str(N))
        print('Cyl Radius = ' + str(rad_cyl))
        print('Bead Radius = ' + str(rad_bead))
    
    print('###### Final Values: ')
    print('N = ' + str(N))
    print('Cyl Radius = ' + str(rad_cyl))
    print('Bead Radius = ' + str(rad_bead))
    
    return N, rad_cyl, rad_bead

#Takes in a trajectory and returns the MSD curves for every bead as a matrix organizedby one bead per row.
#dt_perFrame = time between capturedconformations. For a time step of .01 and block size of 10, dt_perFrame=.1
#COM: If true, subtracts off the center of mass motion of the whole chromosome.
def calc_bead_MSDs(trajFile, max_tau = 100, dt_perFrame = .1, chronum=7, delta_tau = .1, eqFrames = 0, COM=True, skipnum = 5):
    
    MSD_Matrix = []
    timevec = []
    
    start = time.time()
    chro = h5py.File(trajFile, 'r') # comando para abrir o arquivo
    frames = len(chro.keys()) - 1
    firstkey = list(chro.keys())[0]
    
    keylist = list(chro.keys())
    flabels = np.sort([int(key) for key in keylist[0:len(keylist)-1]])
    
    max_jump = int(max_tau/dt_perFrame)
    jumpjump = int(delta_tau/dt_perFrame)
    
    print('####################################################################')
    print('MSD Calculation for Chromosome: Chro {:} with {:} beads'.format(chronum, (len(chro[firstkey]))))
    
    if (frames-eqFrames)<max_jump:
        print('Error: not enough frames for max jump.  Changing max jump to half number of frames.')
        max_jump = int((frames-eqFrames)//2)

    for delta_frame in range(1, max_jump+1, jumpjump):

        ctr = 0
        meanDists = np.zeros(len(chro[firstkey])).reshape(-1,1)
        for i in range(eqFrames, frames-delta_frame-1, skipnum):
            XYZ_0 = np.asarray(chro[str(flabels[i])])
            XYZ_F = np.asarray(chro[str(flabels[i+delta_frame])])
            
            diffs = XYZ_F-XYZ_0
            if COM:
                COMDiff = np.subtract(np.mean(XYZ_F, axis=0), np.mean(XYZ_0, axis = 0))
                diffs = np.subtract(diffs, COMDiff)
                
            curDists = np.matmul(np.multiply(diffs, diffs), np.ones(3).reshape(-1,1))
            meanDists = np.add(meanDists, curDists)
            ctr+=1
            
        timevec.append(delta_frame*dt_perFrame)
            
        if (delta_frame-1)%(max_jump//10)==0:
            print("Calculated MSDs for "+ str(dt_perFrame*delta_frame)+r' $/tau$')
            
    
        meanDists = np.array(np.divide(meanDists, ctr))
        MSD_Matrix.append(meanDists)
    
    #print(MSD_Matrix)
    MSD_Matrix = np.hstack(MSD_Matrix)
    
    chro.close()
    
    end     = time.time()
    elapsed = end - start
    
    print('Total frames: {:}'.format(frames))
    print('############################################################')
    
    return MSD_Matrix, timevec

def calcPvDTimeBins(fileName, mu, r_cut, eqFrames=0, numGroups = 1, chronum = 7):
    
    start = time.time()
    chro = h5py.File(fileName, 'r') # comando para abrir o arquivo
    frames = len(chro.keys()) - 1
    firstkey = list(chro.keys())[0]
    
    print('####################################################################')
    print('Making PvD of Chromosome: Chro {:} with {:} beads'.format(chronum, (len(chro[firstkey]))))
    
    l = len(chro[firstkey])
    
    keylist = list(chro.keys())
    AllFLabels = np.sort([int(key) for key in keylist[0:len(keylist)-1]])
    print(AllFLabels)
    
    FramesPerGroup = (frames-eqFrames) // numGroups
    
    Time_Group_Dict = {}
    Time_Group_Dict['FPG'] = FramesPerGroup
    
    ticker = eqFrames # index of current frame
    
    for curGroup in range(numGroups):
        
        print('On Group: ' + str(curGroup))
        
        PvD = np.zeros((len(chro[firstkey])))
        
        for p in range(FramesPerGroup):
            flabel = AllFLabels[ticker]

            if (ticker) % 500 == 0:
                print('Analyzing Frame: ' + str(ticker) + ' out of ' + str(frames))

            XYZ = np.asarray(chro[str(flabel)])
            
            D = distance.cdist(XYZ, XYZ, 'euclidean')
            D[D<=1.0] = 0.0
            P = 0.5 * (1.0 + np.tanh(mu * (r_cut - D)))
            PvD+=np.array([np.divide(np.trace(P, i), (l-i)) for i in range(l)])
            ticker+=1
            
        PvD = np.divide(PvD, FramesPerGroup)    
        Time_Group_Dict[curGroup] = PvD.tolist()

    chro.close()

    end     = time.time()
    elapsed = end - start

    print('Total frames: {:}'.format(ticker))
    print('Ran in %.3f sec' % elapsed)
    print('############################################################')
    return Time_Group_Dict
