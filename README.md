# PLUMED patch for gREST-enabled TAMD and String Method in Gromacs
TAMD is an enhanced sampling method to explore free energy landscape. On the other hand, the string method has the ability to find minimum free energy paths connecting landmarks on the free energy surface. 

The aim of this project is to implement TAMD/SM in PLUMED+GROMACS. The method also supports the implementation of solute tempering methods such as REST2, and gREST.  
Please cite the following article if using this code:

**PLUMED INSTALLATION:**
1. Create a new directory "mfep" to PLUMED home directory/src/
2. Download and copy the MFEP.cpp and sm.h to mfep directory
3. To enable MFEP, configure PLUMED with --enable-modules=mfep
4. compile Plumed as usual

**GROMACS INSTALLATION**
1. Copy the patch_resm_gmx_2019_6.patch to GROMACS home directory
2. After patching with PLUMED, patch the file by executing:
   patch -p0 < patch_resm_gmx_2019_6.patch
3. Compile GROMACS as usual.   

With succesfull installation, the programs are now ready to launch string method calculation. 
The following input is used for SM calculation in 2D dihedral CV space

**#Plumed input file:**

MFEP ...

NIMAGES=8                    # Number of Images

LABEL=mfep                   # label for code section

ARG=torsion1,torsion2        # Collective varibles 

FICT=0.0,0.0                 # Provide inital value for CV, with 0 it will pick the CV values directly from simulation.

KAPPA=100,100                # Force constant

GAMMABAR=100                 # Friction parameter; with gammabar=0, mean force calculations can be achieved

INTERVAL=200                 # Steps for force accumulation

INTERPOLATION=LINEAR         # Interpolation scheme, use TAMD for TAMD calculations

FREQ=1                       # Frequency for dumping the data

TEMP=0.0                     # Temperature for finite tempeature string/TAMD

FICT_MAX=3.1415,3.1415       # Upper CV boundaries

FICT_MIN=-3.1415,-3.1415     # Lower CV boundaries

ATINDCV=1,2,3,4,2,3,4,5      # Local index assigned to atoms/groups/centers in CV

MCENTER=1,1,1,1,1,1,1,1      # Mass for each center

EVOLVE_ENDS=11               # Evolve both ends=11, first end=10, last end=01, Fix the end=00 

... MFEP


