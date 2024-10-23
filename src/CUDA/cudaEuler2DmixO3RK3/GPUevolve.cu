#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <cuda_runtime.h>

#include "GPUevolve.h"
#include "cudaLimiter.h"
#include "cudaRiemannSolver.h"
#include "cudaEulerBCs.h"
#include "cudaFieldOperation.h"


void setDeviceFieldData(
	fieldPointer& devicePtr,
	fieldPointer& hostPtr,
	const int cellsNum,
	const int facesNum,
	const int patchesNum,
	const int totalBoundaryFacesNum,
	const int maxStencilSize,
	const int maxCompactStencilSize,
	const int maxLocalBlockStencilSize
)
{
	dim3 blockDim(BLOCKDIM);
	dim3 gridDim((cellsNum + blockDim.x - 1)/blockDim.x);

	cudaMalloc((void**)&devicePtr.CELL, sizeof(meshCellData) * (cellsNum + totalBoundaryFacesNum));
	cudaMemcpy(devicePtr.CELL, hostPtr.CELL, sizeof(meshCellData) * (cellsNum + totalBoundaryFacesNum), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.FACE, sizeof(meshFaceData) * (facesNum + totalBoundaryFacesNum));
	cudaMemcpy(devicePtr.FACE, hostPtr.FACE, sizeof(meshFaceData) * (facesNum + totalBoundaryFacesNum), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.cellVolume, sizeof(double) * cellsNum);
	cudaMemcpy(devicePtr.cellVolume, hostPtr.cellVolume, sizeof(double) * cellsNum, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.FD, sizeof(basicFieldData) * (cellsNum + totalBoundaryFacesNum));
	cudaMemcpy(devicePtr.FD, hostPtr.FD, sizeof(basicFieldData) * (cellsNum + totalBoundaryFacesNum), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.limiter, sizeof(limiterFieldData) * cellsNum);

	cudaMalloc((void**)&devicePtr.shockIndicator, sizeof(int8_t) * (cellsNum + totalBoundaryFacesNum));
	cudaMemcpy(devicePtr.shockIndicator, hostPtr.shockIndicator, sizeof(int8_t) * (cellsNum + totalBoundaryFacesNum), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.cFD, sizeof(conservedFieldData) * cellsNum);
	cudaMemcpy(devicePtr.cFD, hostPtr.cFD, sizeof(conservedFieldData) * cellsNum, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.cFDold, sizeof(conservedFieldData) * cellsNum);
	cudaMemcpy(devicePtr.cFDold, hostPtr.cFD, sizeof(conservedFieldData) * cellsNum, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.gradFD, sizeof(gradientFieldData) * cellsNum);

	cudaMalloc((void**)&devicePtr.Flux, sizeof(basicFluxData) * (facesNum + totalBoundaryFacesNum));
	cudaMemcpy(devicePtr.Flux, hostPtr.Flux, sizeof(basicFluxData) * (facesNum + totalBoundaryFacesNum), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.Res, sizeof(residualFieldData) * cellsNum);
	cudaMemcpy(devicePtr.Res, devicePtr.Res, sizeof(residualFieldData) * cellsNum, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.neighbour, sizeof(int) * facesNum*2);
	cudaMemcpy(devicePtr.neighbour, hostPtr.neighbour, sizeof(int) * facesNum*2, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.boundaryFacesNum, sizeof(int) * patchesNum);
	cudaMemcpy(devicePtr.boundaryFacesNum, hostPtr.boundaryFacesNum, sizeof(int) * patchesNum, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.boundaryFaceNeiLabel, sizeof(int) * totalBoundaryFacesNum);
	cudaMemcpy(devicePtr.boundaryFaceNeiLabel, hostPtr.boundaryFaceNeiLabel, sizeof(int) * totalBoundaryFacesNum, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.boundaryFacesType, sizeof(uint8_t) * totalBoundaryFacesNum);
	cudaMemcpy(devicePtr.boundaryFacesType, hostPtr.boundaryFacesType, sizeof(uint8_t) * totalBoundaryFacesNum, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.stencilSize, sizeof(uint8_t) * cellsNum);
	cudaMemcpy(devicePtr.stencilSize, hostPtr.stencilSize, sizeof(uint8_t) * cellsNum, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.compactStencilSize, sizeof(uint8_t) * cellsNum);
	cudaMemcpy(devicePtr.compactStencilSize, hostPtr.compactStencilSize, sizeof(uint8_t) * cellsNum, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.RBFbasis, sizeof(float) * 2*(maxStencilSize+1)*facesNum);
	cudaMemcpy(devicePtr.RBFbasis, hostPtr.RBFbasis, sizeof(float) * 2*(maxStencilSize+1)*facesNum, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.faceFD, sizeof(basicFieldData) * 2*facesNum);

	blockDim.x = BLOCKDIM;
	gridDim.x = (cellsNum + blockDim.x - 1)/blockDim.x;

	cudaMalloc((void**)&devicePtr.extendStencilSize, sizeof(uint16_t) * gridDim.x);
	cudaMemcpy(devicePtr.extendStencilSize, hostPtr.extendStencilSize, sizeof(uint16_t) * gridDim.x, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.extendStencil, sizeof(int) * (maxLocalBlockStencilSize - blockDim.x)*gridDim.x);
	cudaMemcpy(devicePtr.extendStencil, hostPtr.extendStencil, sizeof(int) * (maxLocalBlockStencilSize - blockDim.x)*gridDim.x, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.localStencil, sizeof(uint16_t) * maxStencilSize*cellsNum);
	cudaMemcpy(devicePtr.localStencil, hostPtr.localStencil, sizeof(uint16_t) * maxStencilSize*cellsNum, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.matrix, sizeof(float) * 2*maxCompactStencilSize*cellsNum);

	cudaMalloc((void**)&devicePtr.cellFaces, sizeof(int) * maxCompactStencilSize*cellsNum);
	cudaMemcpy(devicePtr.cellFaces, hostPtr.cellFaces, sizeof(int) * maxCompactStencilSize*cellsNum, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devicePtr.faceDirection, sizeof(int8_t) * maxCompactStencilSize*cellsNum);
	cudaMemcpy(devicePtr.faceDirection, hostPtr.faceDirection, sizeof(int8_t) * maxCompactStencilSize*cellsNum, cudaMemcpyHostToDevice);

	blockDim.x = 1024;
	gridDim.x = (facesNum + blockDim.x - 1)/blockDim.x;
	hostPtr.minDeltaT = (double*)malloc(sizeof(double)*gridDim.x);
	cudaMalloc((void**)&devicePtr.minDeltaT, sizeof(double)*gridDim.x);

	printf("evaluating least square inverse matrix...");
	blockDim.x = BLOCKDIM;
	gridDim.x = (cellsNum + blockDim.x - 1)/blockDim.x;
	const size_t smSize = sizeof(double) * maxLocalBlockStencilSize*2;
	calcLeastSquareMatrix<<<gridDim, blockDim, smSize>>>(
		devicePtr.matrix, devicePtr.localStencil, devicePtr.compactStencilSize,
		devicePtr.CELL, devicePtr.extendStencil, devicePtr.extendStencilSize,
		cellsNum, maxStencilSize, maxCompactStencilSize, maxLocalBlockStencilSize
	);
	printf("complete!\n");

	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess) {
		printf("setDeviceFieldData error! %s\n", cudaGetErrorString(err));
		std::exit(-1);
	}
}


void freeFieldData(fieldPointer& devicePtr, fieldPointer& hostPtr)
{
	cudaFree(devicePtr.CELL);
	cudaFree(devicePtr.FACE);
	cudaFree(devicePtr.cellVolume);
	cudaFree(devicePtr.FD);
	cudaFree(devicePtr.cFD);
	cudaFree(devicePtr.cFDold);
	cudaFree(devicePtr.gradFD);
	cudaFree(devicePtr.Flux);
	cudaFree(devicePtr.Res);
	cudaFree(devicePtr.neighbour);
	cudaFree(devicePtr.boundaryFacesNum);
	cudaFree(devicePtr.boundaryFaceNeiLabel);
	cudaFree(devicePtr.boundaryFacesType);
	cudaFree(devicePtr.stencilSize);
	cudaFree(devicePtr.extendStencilSize);
	cudaFree(devicePtr.extendStencil);
	cudaFree(devicePtr.localStencil);
	cudaFree(devicePtr.matrix);
	cudaFree(devicePtr.cellFaces);
	cudaFree(devicePtr.minDeltaT);
}


double adjustTimeStep(
	fieldPointer& devicePtr,
	fieldPointer& hostPtr,
	const int facesNum,
	const double R,
	const double Cv,
	const double CFL
)
{
	dim3 blockDim(1024);
	dim3 gridDim((facesNum + blockDim.x - 1)/blockDim.x);
	setDeltaT<<<gridDim, blockDim>>>(
		devicePtr.minDeltaT, devicePtr.FD,
		devicePtr.CELL, devicePtr.FACE, devicePtr.neighbour,
		facesNum, R, Cv, CFL
	);
	cudaDeviceSynchronize();

	cudaMemcpy(hostPtr.minDeltaT, devicePtr.minDeltaT, sizeof(double)*gridDim.x, cudaMemcpyDeviceToHost);

	double min_deltaT = 1e5;
	for(unsigned i = 0; i < gridDim.x; ++i)
	{
		min_deltaT = min_deltaT > hostPtr.minDeltaT[i] ? hostPtr.minDeltaT[i] : min_deltaT;
	}

	return min_deltaT;
}


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
	const int maxLocalBlockStencilSize
)
{
	const double gamma = R/Cv + 1.0;
	const double beta1[3] = {1.0, 0.75, 0.333333};
	const double beta2[3] = {0.0, 0.25, 0.666667};
	const double beta3[3] = {1.0, 0.25, 0.666667};

	dim3 blockDim(BLOCKDIM);
	dim3 gridDim((cellsNum + blockDim.x - 1)/blockDim.x);

	cudaError_t err;

	// three stages SSP Runge-Kutta time evolution
	for(unsigned i = 0; i < 3; ++i)
	{
		blockDim.x = BLOCKDIM;
		gridDim.x = (cellsNum + blockDim.x - 1)/blockDim.x;
		size_t sharedMemSize = sizeof(double)*maxLocalBlockStencilSize*2
			+ sizeof(float)*maxLocalBlockStencilSize*4
			+ sizeof(int)*maxLocalBlockStencilSize
			+ sizeof(uint16_t)*blockDim.x*maxStencilSize
			+ sizeof(uint8_t)*blockDim.x*2;
		reconstruct<<<gridDim, blockDim, sharedMemSize>>>(
			devicePtr.faceFD, devicePtr.gradFD, devicePtr.limiter, devicePtr.shockIndicator,
			devicePtr.FD, devicePtr.matrix, devicePtr.RBFbasis, devicePtr.CELL, devicePtr.FACE, devicePtr.cellFaces,
			devicePtr.localStencil, devicePtr.stencilSize, devicePtr.compactStencilSize,
			devicePtr.extendStencil, devicePtr.extendStencilSize,
			cellsNum, maxStencilSize, maxCompactStencilSize, maxLocalBlockStencilSize
		);

		sharedMemSize = sizeof(double)*blockDim.x*4
			+ sizeof(float)*blockDim.x*2
			+ sizeof(int)*maxLocalBlockStencilSize
			+ sizeof(uint16_t)*blockDim.x*maxStencilSize
			+ sizeof(uint8_t)*blockDim.x
			+ sizeof(int8_t)*maxLocalBlockStencilSize;
		BVDindicator<<<gridDim, blockDim, sharedMemSize>>>(
			devicePtr.faceFD, devicePtr.gradFD, devicePtr.shockIndicator, devicePtr.limiter,
			devicePtr.FD, devicePtr.CELL, devicePtr.FACE, devicePtr.cellFaces,
			devicePtr.localStencil, devicePtr.stencilSize, devicePtr.compactStencilSize,
			devicePtr.extendStencil, devicePtr.extendStencilSize,
			cellsNum, maxStencilSize, maxCompactStencilSize, maxLocalBlockStencilSize
		);

		blockDim.x = BLOCKDIM;
		gridDim.x = (facesNum + blockDim.x - 1)/blockDim.x;
		evaluateFlux<<<gridDim, blockDim>>>(
			devicePtr.Flux, devicePtr.FD, devicePtr.gradFD, devicePtr.limiter,
			devicePtr.faceFD, devicePtr.shockIndicator,
			devicePtr.CELL, devicePtr.FACE, devicePtr.neighbour,
			facesNum, cellsNum, totalBoundaryFacesNum, R, Cv
		);

		gridDim.x = (cellsNum + blockDim.x - 1)/blockDim.x;
		evaluateResidual<<<gridDim, blockDim>>>(
			devicePtr.Res, devicePtr.Flux, devicePtr.cellFaces,
			devicePtr.compactStencilSize, devicePtr.faceDirection,
			cellsNum, maxCompactStencilSize
		);

		gridDim.x = (cellsNum + blockDim.x - 1)/blockDim.x;
		updateFieldData<<<gridDim, blockDim>>>(
			devicePtr.FD, devicePtr.cFD, devicePtr.cFDold,
			devicePtr.Res, devicePtr.cellVolume,
			R, gamma, cellsNum, deltaT,
			beta1[i], beta2[i], beta3[i]
		);

		blockDim.x = BLOCKDIM;
		gridDim.x = (totalBoundaryFacesNum + blockDim.x - 1)/blockDim.x;
		updateBoundaryFieldData<<<gridDim, blockDim>>>(
			devicePtr.FD, devicePtr.CELL, devicePtr.FACE,
			devicePtr.boundaryFaceNeiLabel, devicePtr.boundaryFacesType,
			cellsNum, facesNum, totalBoundaryFacesNum,
			R, Cv
		);
	}

	gridDim.x = (cellsNum + blockDim.x - 1)/blockDim.x;
	storeConservedFieldData<<<gridDim, blockDim>>>(devicePtr.cFDold, devicePtr.cFD, cellsNum);

	cudaDeviceSynchronize();

	err = cudaGetLastError();
	if(err != cudaSuccess) {
		printf("GPUevolve error! %s\n", cudaGetErrorString(err));
		std::exit(-1);
	}
}


void copyFieldDataDeviceToHost(
	fieldPointer& hostPtr,
	fieldPointer& devicePtr,
	const int cellsNum,
	const int totalBoundaryFacesNum
)
{
	cudaMemcpy(hostPtr.FD, devicePtr.FD, sizeof(basicFieldData)*(cellsNum + totalBoundaryFacesNum), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostPtr.shockIndicator, devicePtr.shockIndicator, sizeof(int8_t)*(cellsNum + totalBoundaryFacesNum), cudaMemcpyDeviceToHost);
}