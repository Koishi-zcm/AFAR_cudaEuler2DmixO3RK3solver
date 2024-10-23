#ifndef cudaFieldOperation_H
#define cudaFieldOperation_H

#include "basicDataStructure.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do
	{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif


__global__ void setDeltaT(
	double* __restrict__ minDeltaT,
	const basicFieldData* __restrict__ FD,
	const meshCellData* __restrict__ CELL,
	const meshFaceData* __restrict__ FACE,
	const int* __restrict__ neighbour,
	const int facesNum, const double R, const double Cv,
	const double CFL
)
{
	const int faceI = threadIdx.x + blockIdx.x*blockDim.x;

	__shared__ double deltaT[1024];

	deltaT[threadIdx.x] = 1e5;

	if (faceI < facesNum)
	{
		const int own = neighbour[2*faceI];
		const int nei = neighbour[2*faceI+1];

		const double deltaR = hypot(CELL[nei].x - CELL[own].x, CELL[nei].y - CELL[own].y);
		const double fraction = hypot(FACE[faceI].x - CELL[own].x, FACE[faceI].y - CELL[own].y)/deltaR;

		const double Sfx = FACE[faceI].Sfx;
		const double Sfy = FACE[faceI].Sfy;
		const double magSf = FACE[faceI].magSf;

		deltaT[threadIdx.x] = CFL*deltaR*magSf / (
				// face velocity
				(1.0 - fraction)*fabs(FD[own].Ux*Sfx + FD[own].Uy*Sfy) + fraction*fabs(FD[nei].Ux*Sfx + FD[nei].Uy*Sfy)
				// face speed of sound
				+ sqrt( (R/Cv + 1.0)*R*((1.0 - fraction)*FD[own].T + fraction*FD[nei].T) )*magSf
			);

		if (isnan(deltaT[threadIdx.x]))
		{
			printf("own=%d, nei=%d, FD[own].T=%f, FD[nei].T=%f\n", own, nei, FD[own].T, FD[nei].T);
		}
	}

	__syncthreads();

	for(int tid = blockDim.x/2; tid > 0; tid = tid/2) {
		if(threadIdx.x < tid) {
			int lidLeft = threadIdx.x;
			int lidRight = lidLeft + tid;
			deltaT[lidLeft] = deltaT[lidLeft] < deltaT[lidRight] ? deltaT[lidLeft] : deltaT[lidRight];
		}
		
		__syncthreads();
	}

	if(threadIdx.x == 0) {
		minDeltaT[blockIdx.x] = deltaT[0];
	}
}


__global__ void calcLeastSquareMatrix(
	float* __restrict__ matrix,
	const uint16_t* __restrict__ localStencil,
	const uint8_t* __restrict__ compactStencilSize,
	const meshCellData* __restrict__ CELL,
	const int* __restrict__ extendStencil,
	const uint16_t* __restrict__ extendStencilSize,
	const int cellsNum,
	const int maxStencilSize,
	const int maxCompactStencilSize,
	const int maxLocalBlockStencilSize
)
{
	const int celli = threadIdx.x + blockIdx.x*blockDim.x;

	extern __shared__ double sm_calcLeastSquareMatrix[];
	double* x0Stencil = (double*)sm_calcLeastSquareMatrix;
	double* y0Stencil = (double*)&x0Stencil[maxLocalBlockStencilSize];

	if (celli < cellsNum)
	{
		x0Stencil[threadIdx.x] = CELL[celli].x;
		y0Stencil[threadIdx.x] = CELL[celli].y;
	}

	{
		const uint16_t ESS = extendStencilSize[blockIdx.x];
		const uint16_t loops = (ESS + blockDim.x - 1)/blockDim.x;
		for (int i = 0; i < loops; ++i)
		{
			const int id = (threadIdx.x + i*blockDim.x < ESS)*(threadIdx.x + i*blockDim.x);
			const int ii = extendStencil[(maxLocalBlockStencilSize - blockDim.x)*blockIdx.x + id];
			x0Stencil[blockDim.x + id] = CELL[ii].x;
			y0Stencil[blockDim.x + id] = CELL[ii].y;
		}
	}

	__syncthreads();

	if (celli < cellsNum)
	{
		const uint8_t m = compactStencilSize[celli];
		double Sxx = 0.0;
		double Sxy = 0.0;
		double Syy = 0.0;

		for(uint8_t i = 0; i < m; ++i)
		{
			const uint16_t j = localStencil[maxStencilSize*celli + i];
			const double& x0 = x0Stencil[threadIdx.x];
			const double& y0 = y0Stencil[threadIdx.x];
			const double& xj = x0Stencil[j];
			const double& yj = y0Stencil[j];
			Sxx += (xj - x0)*(xj - x0);
			Sxy += (xj - x0)*(yj - y0);
			Syy += (yj - y0)*(yj - y0);
		}

		for(uint8_t i = 0; i < m; ++i)
		{
			const uint16_t j = localStencil[maxStencilSize*celli + i];
			const double& x0 = x0Stencil[threadIdx.x];
			const double& y0 = y0Stencil[threadIdx.x];
			const double& xj = x0Stencil[j];
			const double& yj = y0Stencil[j];
			matrix[2*maxCompactStencilSize*celli + 2*i] = (Syy*(xj - x0) - Sxy*(yj - y0))/(Sxx*Syy - Sxy*Sxy);
			matrix[2*maxCompactStencilSize*celli + 2*i+1] = (-Sxy*(xj - x0) + Sxx*(yj - y0))/(Sxx*Syy - Sxy*Sxy);
		}
	}
}


__global__ void evaluateResidual(
	residualFieldData* __restrict__ Res,
	const basicFluxData* __restrict__ Flux,
	const int* __restrict__ cellFaces,
	const uint8_t* __restrict__ compactStencilSize,
	const int8_t* __restrict__ faceDirection,
	const int cellsNum,
	const int maxCompactStencilSize
)
{
	const int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < cellsNum)
	{
		for (int fi = 0; fi < compactStencilSize[idx]; ++fi)
		{
			const int faceI = cellFaces[maxCompactStencilSize*idx + fi];
			const int8_t direction = faceDirection[maxCompactStencilSize*idx + fi];
			Res[idx].Rrho += direction*Flux[faceI].rhoFlux;
			Res[idx].RrhoUx += direction*Flux[faceI].rhoUxFlux;
			Res[idx].RrhoUy += direction*Flux[faceI].rhoUyFlux;
			Res[idx].RrhoE += direction*Flux[faceI].rhoEFlux;
		}
	}
}


__global__ void updateFieldData(
	basicFieldData* __restrict__ FD,
	conservedFieldData* __restrict__ cFD,
	const conservedFieldData* __restrict__ cFDold,
	residualFieldData* __restrict__ Res,
	const double* __restrict__ cellVolume,
	const double R,
	const double gamma,
	const int cellsNum,
	const double deltaT,
	const double beta1,
	const double beta2,
	const double beta3
)
{
	const int celli = threadIdx.x + blockIdx.x*blockDim.x;

	if (celli < cellsNum)
	{
		const double rho = beta1*cFDold[celli].rho + beta2*cFD[celli].rho + beta3*deltaT*(-Res[celli].Rrho/cellVolume[celli]);
		const double rhoUx = beta1*cFDold[celli].rhoUx + beta2*cFD[celli].rhoUx + beta3*deltaT*(-Res[celli].RrhoUx/cellVolume[celli]);
		const double rhoUy = beta1*cFDold[celli].rhoUy + beta2*cFD[celli].rhoUy + beta3*deltaT*(-Res[celli].RrhoUy/cellVolume[celli]);
		const double rhoE = beta1*cFDold[celli].rhoE + beta2*cFD[celli].rhoE + beta3*deltaT*(-Res[celli].RrhoE/cellVolume[celli]);

		cFD[celli].rho = rho;
		cFD[celli].rhoUx = rhoUx;
		cFD[celli].rhoUy = rhoUy;
		cFD[celli].rhoE = rhoE;

		FD[celli].p = (gamma - 1.0)*(rhoE - 0.5*(rhoUx*rhoUx + rhoUy*rhoUy)/rho);
		FD[celli].Ux = rhoUx/rho;
		FD[celli].Uy = rhoUy/rho;
		FD[celli].T = FD[celli].p/(R*rho);

		// reset residual to zero
		Res[celli].Rrho = 0.0;
		Res[celli].RrhoUx = 0.0;
		Res[celli].RrhoUy = 0.0;
		Res[celli].RrhoE = 0.0;
	}
}


__global__ void storeConservedFieldData(
	conservedFieldData* __restrict__ cFDold,
	const conservedFieldData* __restrict__ cFD,
	const int cellsNum
)
{
	const int celli = threadIdx.x + blockIdx.x*blockDim.x;

	if (celli < cellsNum)
	{
		cFDold[celli].rho = cFD[celli].rho;
		cFDold[celli].rhoUx = cFD[celli].rhoUx;
		cFDold[celli].rhoUy = cFD[celli].rhoUy;
		cFDold[celli].rhoE = cFD[celli].rhoE;
	}
}

#endif