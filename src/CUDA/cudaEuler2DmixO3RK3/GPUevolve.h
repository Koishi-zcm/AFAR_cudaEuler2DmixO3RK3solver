#ifndef GPUevolve_H
#define GPUevolve_H

#include "basicDataStructure.h"

extern "C" void setDeviceFieldData(
	fieldPointer& devicePtr,
	fieldPointer& hostPtr,
	const int cellsNum,
	const int facesNum,
	const int patchesNum,
	const int totalBoundaryFacesNum,
	const int maxStencilSize,
	const int maxCompactStencilSize,
	const int maxLocalBlockStencilSize
);

extern "C" void freeFieldData(fieldPointer& devicePtr, fieldPointer& hostPtr);

extern "C" double adjustTimeStep(
	fieldPointer& devicePtr,
	fieldPointer& hostPtr,
	const int facesNum,
	const double R,
	const double Cv,
	const double CFL
);

extern "C" void GPUevolve(
	fieldPointer& devicePtr,
	fieldPointer& hostPtr,
	const double R,
	const double Cv,
	const double deltaT,
	const int cellsNum,
	const int facesNum,
	const int totalBoundaryFacesNum,
	const int maxStencilSize,
	const int maxCompactStencilSize,
	const int maxLocalBlockStencilSize
);


extern "C" void copyFieldDataDeviceToHost(
	fieldPointer& hostPtr,
	fieldPointer& devicePtr,
	const int cellsNum,
	const int totalBoundaryFacesNum
);

#endif