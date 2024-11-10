FOAM_SRC = /home/zcm21/OpenFOAM/OpenFOAM-10/src
USER_LIB = /home/zcm21/OpenFOAM/zcm21-10/platforms/linux64GccDPInt32Opt/lib

NAME = cudaEuler2DmixO3RK3

FLAG1 = -std=c++14 -m64 -Dlinux64 -DWM_ARCH_OPTION=64 -DWM_DP -DWM_LABEL_SIZE=32 \
        -Wall -Wextra -Wold-style-cast -Wnon-virtual-dtor -Wno-unused-parameter \
        -Wno-invalid-offsetof -Wno-attributes -O3  -DNoRepository -ftemplate-depth-100 -fPIC
FLAG2 = $(FLAG1) -fuse-ld=bfd -shared -Xlinker --add-needed -Xlinker --no-as-needed

LINK = -I$(FOAM_SRC)/finiteVolume/lnInclude \
        -I$(FOAM_SRC)/physicalProperties/lnInclude \
        -I$(FOAM_SRC)/thermophysicalModels/basic/lnInclude \
        -I$(FOAM_SRC)/thermophysicalModels/specie/lnInclude \
        -I$(FOAM_SRC)/MomentumTransportModels/momentumTransportModels/lnInclude \
        -I$(FOAM_SRC)/MomentumTransportModels/compressible/lnInclude \
        -I$(FOAM_SRC)/ThermophysicalTransportModels/lnInclude \
        -I$(FOAM_SRC)/meshTools/lnInclude \
        -IlnInclude \
        -I. \
        -I../BCs/2D \
        -I/home/zcm21/OpenFOAM/OpenFOAM-10/src/OpenFOAM/lnInclude \
        -I/home/zcm21/OpenFOAM/OpenFOAM-10/src/OSspecific/POSIX/lnInclude \
        -I/home/zcm21/OpenFOAM/zcm21-10/src/gaussQuadrature/lnInclude

LIB = -L/home/zcm21/OpenFOAM/OpenFOAM-10/platforms/linux64GccDPInt32Opt/lib \
      -L/home/zcm21/OpenFOAM/zcm21-10/platforms/linux64GccDPInt32Opt/lib \
        -lfiniteVolume -lfvModels -lfvConstraints -lfluidThermophysicalModels -lspecie \
        -lmomentumTransportModels -lcompressibleMomentumTransportModels -lthermophysicalTransportModels \
        -lmeshTools -lOpenFOAM -ldl -lm -lgaussQuadrature

default:
	g++ $(FLAG1) $(LINK) -c basicCUDAsolver.cpp -o Make/linux64GccDPInt32Opt/basicCUDAsolver.o
	nvcc -O3 -Xcompiler '-fPIC' $(LINK) -dc GPUevolve.cu -o Make/linux64GccDPInt32Opt/GPUevolve.o
	nvcc -O3 -Xcompiler '-fPIC' $(LINK) -dlink Make/linux64GccDPInt32Opt/GPUevolve.o -o Make/linux64GccDPInt32Opt/link.o
	g++ $(FLAG2) -lcudart $(LINK) $(LIB) \
	Make/linux64GccDPInt32Opt/basicCUDAsolver.o \
	Make/linux64GccDPInt32Opt/GPUevolve.o \
	Make/linux64GccDPInt32Opt/link.o \
	-o $(USER_LIB)/lib$(NAME).so

.PHONY:lnInclude clean
lnInclude:
	mkdir ./lnInclude/
	@
	@for file in $(wildcard *.cpp); do \
		ln -s -f $$PWD/$$file ./lnInclude/$$file; \
	done

	@
	@for file in $(wildcard *.h); do \
		ln -s -f $$PWD/$$file ./lnInclude/$$file; \
	done

clean:
	rm -rf ./lnInclude/
	rm -rf *.o
	rm $(USER_LIB)/lib$(NAME).so
