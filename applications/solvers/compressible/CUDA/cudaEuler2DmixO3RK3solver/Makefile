NAME = cudaEuler2DmixO3RK3

FOAM_SRC = /home/zcm21/OpenFOAM/OpenFOAM-10/src
USER_BIN = /home/zcm21/OpenFOAM/zcm21-10/platforms/linux64GccDPInt32Opt/bin

FLAG1 = -std=c++14 -m64 -Dlinux64 -DWM_ARCH_OPTION=64 -DWM_DP -DWM_LABEL_SIZE=32 -O3 -DNoRepository -fPIC
FLAG2 = $(FLAG1) -fuse-ld=bfd -Xlinker --add-needed -Xlinker --no-as-needed

LINK = -I$(FOAM_SRC)/finiteVolume/lnInclude \
	-I$(FOAM_SRC)/physicalProperties/lnInclude \
	-I$(FOAM_SRC)/thermophysicalModels/basic/lnInclude \
	-I$(FOAM_SRC)/thermophysicalModels/specie/lnInclude \
	-I$(FOAM_SRC)/MomentumTransportModels/momentumTransportModels/lnInclude \
	-I$(FOAM_SRC)/MomentumTransportModels/compressible/lnInclude \
	-I$(FOAM_SRC)/ThermophysicalTransportModels/lnInclude \
	-I$(FOAM_SRC)/meshTools/lnInclude \
	-I/home/zcm21/OpenFOAM/zcm21-10/src/CUDA/Euler/$(NAME)/lnInclude \
	-IlnInclude \
	-I. \
	-I$(FOAM_SRC)/OpenFOAM/lnInclude \
	-I$(FOAM_SRC)/OSspecific/POSIX/lnInclude

LIB = -L/home/zcm21/OpenFOAM/OpenFOAM-10/platforms/linux64GccDPInt32Opt/lib \
      -L/home/zcm21/OpenFOAM/zcm21-10/platforms/linux64GccDPInt32Opt/lib \
	-lfiniteVolume -lfvModels -lfvConstraints -lfluidThermophysicalModels -lspecie \
	-lmomentumTransportModels -lcompressibleMomentumTransportModels -lthermophysicalTransportModels \
	-lmeshTools -lOpenFOAM -ldl -lm \
	-l$(NAME)

fullCompile:
	g++ $(FLAG1) $(LINK) -c $(NAME)solver.cpp -o Make/linux64GccDPInt32Opt/$(NAME)solver.o
	g++ $(FLAG2) -lcudart -lcudadevrt $(LINK) $(LIB) Make/linux64GccDPInt32Opt/$(NAME)solver.o -o $(USER_BIN)/$(NAME)solver
	
clean:
	rm -rf ./Make/linux64GccDPInt32Opt/*.o $(USER_BIN)/$(NAME)solver
