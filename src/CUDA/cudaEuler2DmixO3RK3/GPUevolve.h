#ifndef GPUevolve_H
#define GPUevolve_H

#include "basicDataStructure.h"

void setDeviceFieldData(
	fieldPointer& devicePtr,
	fieldPointer& hostPtr,
	const int cellsNum,
	const int facesNum,
	const int totalBoundaryFacesNum
);

void setDeviceFieldData(
	fieldPointer& devicePtr,
	fieldPointer& hostPtr,
	const int cellsNum,
	const int facesNum,
	const int patchesNum,
	const int totalBoundaryFacesNum,
	const int maxStencilSize,
	const int maxCompactStencilSize,
	const int maxLocalBlockStencilSize,
	const int maxCompactLocalBlockStencilSize
);

void freeFieldData(fieldPointer& devicePtr, fieldPointer& hostPtr);

double adjustTimeStep(
	fieldPointer& devicePtr,
	fieldPointer& hostPtr,
	const int facesNum,
	const double R,
	const double Cv,
	const double CFL
);

void GPUevolve(
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
	const int maxLocalBlockStencilSize,
	const int maxCompactLocalBlockStencilSize
);


void copyFieldDataDeviceToHost(
	fieldPointer& hostPtr,
	fieldPointer& devicePtr,
	const int cellsNum,
	const int totalBoundaryFacesNum
);

#endif