# AFAR_cudaEuler2DmixO3RK3solver

	Adaptive Fluxes and Adaptive Reconstructions (AFAR) method implementation for Euler equations.
	This is a sample code to illustrate the basic idea of AFAR, and it is a single-GPU solver
	based on CUDA framework.

## Installation

	The code is based on the mesh and file I/O system in OpenFOAM and is only tested with
	OpenFOAM-10 and CUDA 11.5. It is required to build a USER directory in OpenFOAM and it
	should be organized as the standard file organization in OpenFOAM-10.
	
	Makefiles are provided here but they are used on the test device. To make it work on
	other devices, please follow the following steps to compile the code:
	
STEP1:
	
	Place the provided src directory and applications directory in your USER directory in
	OpenFOAM.
	
	e.g.:
	/home/zcm21/OpenFOAM/zcm21-10/src
	/home/zcm21/OpenFOAM/zcm21-10/applications
	
STEP2:
	
	Enter src/gaussQuadrature directory, check the Make/options file if you are working with
	other versions of OpenFOAM. Use wmake in OpenFOAM to compile d2GaussQuadrature lib.
	
	e.g.:
	cd /home/zcm21/OpenFOAM/zcm21-10/src/gaussQuadrature/
	wmake
	
	By now you should have libgaussQuadrature.so in your USER_LIB. You can check this in your
	own path.
	
	e.g.:
	/home/zcm21/OpenFOAM/zcm21-10/platforms/linux64GccDPInt32Opt/lib/
	
STEP3:
	Enter src/CUDA/cudaEuler2DmixO3RK3 directory, check the Makefile and replace the FOAM_SRC,
	USER_LIB and some LINK into your correct paths. For other versions of OpenFOAM you may also
	need to remove some unnecessary LIB like -lmomentumTransportModels. After correcting all the
	paths, use make lnInclude to create file links and make default for solver lib.
	
	e.g.:
	cd /home/zcm21/OpenFOAM/zcm21-10/src/CUDA/cudaEuler2DmixO3RK3/
	make lnInclude
	make default
	
	By now you should have libcudaEuler2DmixO3RK3.so in your USER_LIB.
	
STEP4:
	Enter applications/solvers/compressible/CUDA/cudaEuler2DmixO3RK3solver directory, check the
	Makefile and replace the FOAM_SRC, USER_LIB and some LINK and LIB into your correct paths.
	For other versions of OpenFOAM you may also need to remove some unnecessary LIB like
	-lmomentumTransportModels. After correcting all the paths, use make fullCompile for the solver.
	
	e.g.:
	cd /home/zcm21/OpenFOAM/zcm21-10/applications/solvers/compressible/CUDA/cudaEuler2DmixO3RK3solver/
	make fullCompile
	
	By now you should have cudaEuler2DmixO3RK3solver in your USER_BIN. You can check this in your
	own path.
	
	e.g.:
	/home/zcm21/OpenFOAM/zcm21-10/platforms/linux64GccDPInt32Opt/bin/
	
	And the installation is completed.
	
## Usage

	The case setting is the same as the general OpenFOAM case setting. The setting of cudaEuler2DmixO3RK3solver
	is completed in the fvSchemes file. In this file, you should set gaussQuadrature and CUDA dict.
	The boundary conditions is set in the subDict boundary in the CUDA dict. The label for each patch
	should be consistent with the src/CUDA/BCs/cudaEulerBCs.h. Note that the OpenFOAM-style boundary
	setting is also required for field initialization. Besides, the 2D case mesh should always set
	in x-y plane for correct behavior of the solver.
	
	An example case cornerFlow is provided to show the basic setting of cudaEuler2DmixO3RK3solver. The simulation
	steps:
	1. use blockMesh to create computational domain.
	2. always remember to run renumberMesh -overwrite to reorder the cell labels.
	3. run ./cudaEuler2DmixO3RK3solver
	Paraview is suggested for data visualization.
