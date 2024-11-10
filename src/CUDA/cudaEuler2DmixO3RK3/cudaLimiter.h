#ifndef cudaLimiter_H
#define cudaLimiter_H

#define SMALL 1e-5f
#define VSMALL 1e-10f
#define BVD_FACTOR 0.5f
#define VALID_LIM_FACTOR 0.9f

__device__ __forceinline__ float ToMinmod(const float& yy)
{
	return fminf(yy, 1.0f);
}

__device__ __forceinline__ float ToBarthJespersen(const float& yy)
{
	return fminf(2.0f*yy, 1.0f);
}

__device__ __forceinline__ float ToVenkatakrishnan(const float& yy)
{
	return fminf((yy*yy + yy + VSMALL)/(yy*yy + 0.5f + 0.5f*yy + VSMALL), 1.0f);
}

__device__ __forceinline__ float ToSuperbee(const float& yy)
{
	return fmaxf(fminf(2.0f*yy, 1.0f), yy);
}

__device__ __forceinline__ float transFormLimiter(const float& lim, const bool& isShock)
{
	return isShock*(ToVenkatakrishnan(lim)) + (1 - isShock)*(ToBarthJespersen(lim));
}


__device__ void limiterProcess(
	float& limiter, float& phif, float& d1, float& d2, float& yy,
	const float& phic, const float& phiMax, const float& phiMin,
	const double& gradPhix, const double& gradPhiy,
	const double& dx, const double& dy, const float& eps2
)
{
	phif = phic;
	phif += gradPhix*dx + gradPhiy*dy;
	d2 = phif - phic;
	d1 = (phif < phic)*(phiMin - phic) + (phif >= phic)*(phiMax - phic);
	yy = fmin(2.0f, 0.5f*d1/(d2 + (d2 >= 0.0f)*VSMALL - (d2 < 0.0f)*VSMALL));
	const bool valid = (fabs(d2) >= SMALL);
	yy = valid*yy + (1 - valid)*1.0f;
	limiter = fmin(limiter, yy);
}


__global__ void reconstruct(
	basicFieldData* __restrict__ faceFD,
	gradientFieldData* __restrict__ gradFD,
	limiterFieldData* __restrict__ limiter,
	int8_t* __restrict__ shockIndicator,
	const basicFieldData* __restrict__ FD,
	const float* __restrict__ matrix,
	const float* __restrict__ RBFbasis,
	const meshCellData* __restrict__ CELL,
	const meshFaceData* __restrict__ FACE,
	const int* __restrict__ cellFaces,
	const uint16_t* __restrict__ localStencil,
	const uint8_t* __restrict__ stencilSize,
	const uint8_t* __restrict__ compactStencilSize,
	const int* __restrict__ extendStencil,
	const uint16_t* __restrict__ extendStencilSize,
	const uint16_t* __restrict__ compactExtendStencilSize,
	const int cellsNum,
	const int facesNum,
	const int maxStencilSize,
	const int maxCompactStencilSize,
	const int maxLocalBlockStencilSize,
	const int maxCompactLocalBlockStencilSize
)
{
	const int celli = threadIdx.x + blockIdx.x*blockDim.x;

	extern __shared__ double sm_reconstruct[];
	double* x0Stencil = sm_reconstruct;
	double* y0Stencil = (double*)&x0Stencil[maxCompactLocalBlockStencilSize];
	float* pStencil = (float*)&y0Stencil[maxCompactLocalBlockStencilSize];
	float* UxStencil = (float*)&pStencil[maxLocalBlockStencilSize];
	float* UyStencil = (float*)&UxStencil[maxLocalBlockStencilSize];
	float* TStencil = (float*)&UyStencil[maxLocalBlockStencilSize];
	uint16_t* stencil = (uint16_t*)&TStencil[maxLocalBlockStencilSize];
	uint8_t* localStencilSize = (uint8_t*)&stencil[maxStencilSize*blockDim.x];
	uint8_t* localCompactStencilSize = (uint8_t*)&localStencilSize[blockDim.x];

	if (celli < cellsNum)
	{
		localStencilSize[threadIdx.x] = stencilSize[celli];
		localCompactStencilSize[threadIdx.x] = compactStencilSize[celli];
		pStencil[threadIdx.x] = FD[celli].p;
		UxStencil[threadIdx.x] = FD[celli].Ux;
		UyStencil[threadIdx.x] = FD[celli].Uy;
		TStencil[threadIdx.x] = FD[celli].T;
		x0Stencil[threadIdx.x] = CELL[celli].x;
		y0Stencil[threadIdx.x] = CELL[celli].y;

		for (int i = 0; i < localStencilSize[threadIdx.x]; ++i)
		{
			stencil[maxStencilSize*threadIdx.x + i] = localStencil[maxStencilSize*celli + i];
		}
	}


	{
		const uint16_t ESS = extendStencilSize[blockIdx.x];
		const int16_t loops = (ESS + blockDim.x - 1)/blockDim.x;
		for (int i = 0; i < loops; ++i)
		{
			const int id = (threadIdx.x + i*blockDim.x < ESS)*(threadIdx.x + i*blockDim.x);
			const int ii = extendStencil[(maxLocalBlockStencilSize - blockDim.x)*blockIdx.x + id];
			pStencil[blockDim.x + id] = FD[ii].p;
			UxStencil[blockDim.x + id] = FD[ii].Ux;
			UyStencil[blockDim.x + id] = FD[ii].Uy;
			TStencil[blockDim.x + id] = FD[ii].T;
		}
	}

	{
		const uint16_t ESS = compactExtendStencilSize[blockIdx.x];
		const int16_t loops = (ESS + blockDim.x - 1)/blockDim.x;
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
		double tGradPx = 0.0;
		double tGradPy = 0.0;
		double tGradUxx = 0.0;
		double tGradUxy = 0.0;
		double tGradUyx = 0.0;
		double tGradUyy = 0.0;
		double tGradTx = 0.0;
		double tGradTy = 0.0;

		for(int i = 0; i < localCompactStencilSize[threadIdx.x]; ++i)
		{
			const float M1 = matrix[2*maxCompactStencilSize*celli + 2*i];
			const float M2 = matrix[2*maxCompactStencilSize*celli + 2*i+1];
			const uint16_t& j = stencil[maxStencilSize*threadIdx.x + i];

			float dphi = pStencil[j] - pStencil[threadIdx.x];
			tGradPx += M1*dphi;
			tGradPy += M2*dphi;

			dphi = UxStencil[j] - UxStencil[threadIdx.x];
			tGradUxx += M1*dphi;
			tGradUyx += M2*dphi;

			dphi = UyStencil[j] - UyStencil[threadIdx.x];
			tGradUxy += M1*dphi;
			tGradUyy += M2*dphi;

			dphi = TStencil[j] - TStencil[threadIdx.x];
			tGradTx += M1*dphi;
			tGradTy += M2*dphi;
		}

		gradFD[celli].gradPx = tGradPx;
		gradFD[celli].gradPy = tGradPy;
		gradFD[celli].gradUxx = tGradUxx; gradFD[celli].gradUxy = tGradUxy;
		gradFD[celli].gradUyx = tGradUyx; gradFD[celli].gradUyy = tGradUyy;
		gradFD[celli].gradTx = tGradTx;
		gradFD[celli].gradTy = tGradTy;

		float pMax = pStencil[threadIdx.x];
		float pMin = pMax;

		float UxMax = UxStencil[threadIdx.x];
		float UxMin = UxMax;

		float UyMax = UyStencil[threadIdx.x];
		float UyMin = UyMax;
		
		float TMax = TStencil[threadIdx.x];
		float TMin = TMax;

		for(int i = 0; i < localCompactStencilSize[threadIdx.x]; ++i)
		{
			const uint16_t& j = stencil[maxStencilSize*threadIdx.x + i];

			pMax = fmaxf(pStencil[j], pMax);
			pMin = fminf(pStencil[j], pMin);

			UxMax = fmaxf(UxStencil[j], UxMax);
			UxMin = fminf(UxStencil[j], UxMin);

			UyMax = fmaxf(UyStencil[j], UyMax);
			UyMin = fminf(UyStencil[j], UyMin);

			TMax = fmaxf(TStencil[j], TMax);
			TMin = fminf(TStencil[j], TMin);
		}

		const float eps2 = VSMALL;
		if(fabsf(TMax - TMin) < VSMALL)
		{
			limiter[celli].pLimiter = 1.0f;
			limiter[celli].UxLimiter = 1.0f;
			limiter[celli].UyLimiter = 1.0f;
			limiter[celli].TLimiter = 1.0f;

			for(unsigned fi = 0; fi < localCompactStencilSize[threadIdx.x]; ++fi)
			{
				int faceI = cellFaces[maxCompactStencilSize*celli + fi];
				const bool isNei = (faceI < 0);
				faceI = abs(faceI) - 1;

				if (faceI >= facesNum) continue;  // ignore boundary face direction

				const double dx = FACE[faceI].x - x0Stencil[threadIdx.x];
				const double dy = FACE[faceI].y - y0Stencil[threadIdx.x];

				faceFD[2*faceI + isNei].p = (double)(pStencil[threadIdx.x]) + tGradPx*dx + tGradPy*dy;
				faceFD[2*faceI + isNei].Ux = (double)(UxStencil[threadIdx.x]) + tGradUxx*dx + tGradUyx*dy;
				faceFD[2*faceI + isNei].Uy = (double)(UyStencil[threadIdx.x]) + tGradUxy*dx + tGradUyy*dy;
				faceFD[2*faceI + isNei].T = (double)(TStencil[threadIdx.x]) + tGradTx*dx + tGradTy*dy;
			}

			shockIndicator[2*celli] = 0;
		}
		else
		{
			float tmpPlimiter = 2.0f;
			float tmpUxLimiter = 2.0f;
			float tmpUyLimiter = 2.0f;
			float tmpTLimiter = 2.0f;

			float smoothnessValue = 0.0f;

			bool isNotTrouble = true;

			const float& pc = pStencil[threadIdx.x];
			const float& Uxc = UxStencil[threadIdx.x];
			const float& Uyc = UyStencil[threadIdx.x];
			const float& Tc = TStencil[threadIdx.x];

			for(unsigned fi = 0; fi < localCompactStencilSize[threadIdx.x]; ++fi)
			{
				int faceI = cellFaces[maxCompactStencilSize*celli + fi];
				const bool isNei = (faceI < 0);
				faceI = abs(faceI) - 1;

				if (faceI >= facesNum) continue;  // ignore boundary face direction

				const double dx = FACE[faceI].x - x0Stencil[threadIdx.x];
				const double dy = FACE[faceI].y - y0Stencil[threadIdx.x];
				const uint16_t& faceNeiLocalID = stencil[maxStencilSize*threadIdx.x + fi];
				const float deltaR = hypot(x0Stencil[faceNeiLocalID] - x0Stencil[threadIdx.x], y0Stencil[faceNeiLocalID] - y0Stencil[threadIdx.x]);
				const float fraction = sqrt(dx*dx + dy*dy)/deltaR;

				float RBFbasisItem = RBFbasis[2*(maxStencilSize+1)*faceI + (isNei)*(maxStencilSize+1)];
				double pf = RBFbasisItem * pStencil[threadIdx.x];
				double Uxf = RBFbasisItem * UxStencil[threadIdx.x];
				double Uyf = RBFbasisItem * UyStencil[threadIdx.x];
				double Tf = RBFbasisItem * TStencil[threadIdx.x];

				for (int ci = 0; ci < localStencilSize[threadIdx.x]; ++ci)
				{
					const uint16_t& j = stencil[maxStencilSize*threadIdx.x + ci];
					RBFbasisItem = RBFbasis[2*(maxStencilSize+1)*faceI + (isNei)*(maxStencilSize+1) + ci+1];
					pf += RBFbasisItem * pStencil[j];
					Uxf += RBFbasisItem * UxStencil[j];
					Uyf += RBFbasisItem * UyStencil[j];
					Tf += RBFbasisItem * TStencil[j];
				}

				float phif;
				float phiD2;
				float phiD1;
				float phiYY;

				limiterProcess(tmpPlimiter, phif, phiD1, phiD2, phiYY, pc, pMax, pMin, tGradPx, tGradPy, dx, dy, eps2);
				faceFD[2*faceI + isNei].p = (fabsf(pMax - pMin) >= SMALL)*pf + (fabsf(pMax - pMin) < SMALL)*phif;
				isNotTrouble = phif > 0.0f;

				smoothnessValue = fmaxf(smoothnessValue,
					fabsf( (1.0f - fraction)*pStencil[threadIdx.x] 
						+ fraction*pStencil[faceNeiLocalID] - phif
						)/deltaR
						+ (pf <= 0.0f)*1e20f
						+ (phif <= 0.0f)*1e20f
				);

				limiterProcess(tmpUxLimiter, phif, phiD1, phiD2, phiYY, Uxc, UxMax, UxMin, tGradUxx, tGradUyx, dx, dy, eps2);
				faceFD[2*faceI + isNei].Ux = (fabsf(UxMax - UxMin) >= SMALL)*Uxf + (fabsf(UxMax - UxMin) < SMALL)*phif;

				limiterProcess(tmpUyLimiter, phif, phiD1, phiD2, phiYY, Uyc, UyMax, UyMin, tGradUxy, tGradUyy, dx, dy, eps2);
				faceFD[2*faceI + isNei].Uy = (fabsf(UyMax - UyMin) >= SMALL)*Uyf + (fabsf(UyMax - UyMin) < SMALL)*phif;

				limiterProcess(tmpTLimiter, phif, phiD1, phiD2, phiYY, Tc, TMax, TMin, tGradTx, tGradTy, dx, dy, eps2);
				faceFD[2*faceI + isNei].T = (fabsf(TMax - TMin) >= SMALL)*Tf + (fabsf(TMax - TMin) < SMALL)*phif;
				isNotTrouble = phif > 0.0f;
			}

			limiter[celli].pLimiter = fmaxf(tmpPlimiter, 0.0f)*isNotTrouble;
			limiter[celli].UxLimiter = fmaxf(tmpUxLimiter, 0.0f)*isNotTrouble;
			limiter[celli].UyLimiter = fmaxf(tmpUyLimiter, 0.0f)*isNotTrouble;
			limiter[celli].TLimiter = fmaxf(tmpTLimiter, 0.0f)*isNotTrouble;
			smoothnessValue = smoothnessValue/(0.5f*(pMax + pMin)*isNotTrouble + VSMALL);
			shockIndicator[2*celli] = (smoothnessValue > 1.0f);
		}
	}
}


__global__ void BVDindicator(
	basicFieldData* __restrict__ faceFD,
	gradientFieldData* __restrict__ gradFD,
	int8_t* __restrict__ shockIndicator,
	limiterFieldData* __restrict__ limiter,
	const basicFieldData* __restrict__ FD,
	const meshCellData* __restrict__ CELL,
	const meshFaceData* __restrict__ FACE,
	const int* __restrict__ cellFaces,
	const uint16_t* __restrict__ localStencil,
	const uint8_t* __restrict__ stencilSize,
	const uint8_t* __restrict__ compactStencilSize,
	const int* __restrict__ extendStencil,
	const uint16_t* __restrict__ extendStencilSize,
	const int cellsNum,
	const int facesNum,
	const int maxStencilSize,
	const int maxCompactStencilSize,
	const int maxLocalBlockStencilSize
)
{
	const int celli = threadIdx.x + blockIdx.x*blockDim.x;

	extern __shared__ double sm_BVDindicator[];
	double* x0Stencil = sm_BVDindicator;
	double* y0Stencil = (double*)&x0Stencil[blockDim.x];
	double* gradTxStencil = (double*)&y0Stencil[blockDim.x];
	double* gradTyStencil = (double*)&gradTxStencil[blockDim.x];
	float* TStencil = (float*)&gradTyStencil[blockDim.x];
	float* TLimiterStencil = (float*)&TStencil[blockDim.x];
	uint16_t* stencil = (uint16_t*)&TLimiterStencil[blockDim.x];
	uint8_t* localStencilSize = (uint8_t*)&stencil[maxStencilSize*blockDim.x];
	int8_t* shockIndicatorStencil = (int8_t*)&localStencilSize[blockDim.x];

	if (celli < cellsNum)
	{
		TStencil[threadIdx.x] = FD[celli].T;
		gradTxStencil[threadIdx.x] = gradFD[celli].gradTx;
		gradTyStencil[threadIdx.x] = gradFD[celli].gradTy;
		x0Stencil[threadIdx.x] = CELL[celli].x;
		y0Stencil[threadIdx.x] = CELL[celli].y;
		TLimiterStencil[threadIdx.x] = limiter[celli].TLimiter;
		localStencilSize[threadIdx.x] = stencilSize[celli];
		shockIndicatorStencil[threadIdx.x] = shockIndicator[2*celli];

		for (int i = 0; i < localStencilSize[threadIdx.x]; ++i)
		{
			stencil[maxStencilSize*threadIdx.x + i] = localStencil[maxStencilSize*celli + i];
		}
	}

	{
		const uint16_t ESS = extendStencilSize[blockIdx.x];
		const uint16_t loops = (ESS + blockDim.x - 1)/blockDim.x;
		for (int i = 0; i < loops; ++i)
		{
			const int id = (threadIdx.x + i*blockDim.x < ESS)*(threadIdx.x + i*blockDim.x);
			const int ii = extendStencil[(maxLocalBlockStencilSize - blockDim.x)*blockIdx.x + id];
			shockIndicatorStencil[blockDim.x + id] = shockIndicator[2*ii];
		}
	}

	__syncthreads();

	if (celli < cellsNum)
	{
		int BVDtype = 0;
		int bfacesNum = 0;

		const int localCompactStencilSize = compactStencilSize[celli];

		int8_t shockType = shockIndicatorStencil[threadIdx.x];
		int8_t shockType_extend = shockIndicatorStencil[threadIdx.x];

		for (int ci = 0; ci < localCompactStencilSize; ++ci)
		{
			const uint16_t& j = stencil[maxStencilSize*threadIdx.x + ci];
			shockType += shockIndicatorStencil[j];
			shockType_extend = shockType;
		}

		for (int ci = localCompactStencilSize; ci < localStencilSize[threadIdx.x]; ++ci)
		{
			const uint16_t& j = stencil[maxStencilSize*threadIdx.x + ci];
			shockType_extend += shockIndicatorStencil[j];
		}

		// for (int ci = 0; ci < localStencilSize[threadIdx.x]; ++ci)
		// {
		// 	const uint16_t& j = stencil[maxStencilSize*threadIdx.x + ci];
		// 	shockType += shockIndicatorStencil[j];
		// 	shockType_extend += shockIndicatorStencil[j];
		// }

		for(unsigned fi = 0; fi < localCompactStencilSize; ++fi)
		{
			int faceI = cellFaces[maxCompactStencilSize*celli + fi];
			const bool isNei = (faceI < 0);
			faceI = abs(faceI) - 1;
			
			if (faceI >= facesNum)
			{
				bfacesNum += 1;
				if (bfacesNum >= 2)  // use O2 reconstruction on corner cells
				{
					BVDtype = 1;
					break;
				}

				continue;  // ignore boundary face direction, because of no stored O3 value on boundary faces
			}

			const double dx_cur = FACE[faceI].x - x0Stencil[threadIdx.x];
			const double dy_cur = FACE[faceI].y - y0Stencil[threadIdx.x];

			const double TfOwn = faceFD[2*faceI].T;
			const double TfNei = faceFD[2*faceI+1].T;

			const double Tf_cur = isNei*TfNei + (1 - isNei)*TfOwn;
			const double Tf_nei = (1 - isNei)*TfNei + isNei*TfOwn;

			const double lim = ToBarthJespersen(TLimiterStencil[threadIdx.x]);
			const double Tf_cur_TVD = TStencil[threadIdx.x] + lim*(gradTxStencil[threadIdx.x]*dx_cur + gradTyStencil[threadIdx.x]*dy_cur);

			const int typeval = fabs(Tf_cur_TVD - Tf_nei) < BVD_FACTOR*fabs(Tf_cur - Tf_nei);
			const int isNotFlat = fabs(Tf_cur - Tf_nei) > VSMALL;
			const int isLimited = (lim < VALID_LIM_FACTOR) + (shockType_extend > 0);

			BVDtype = max(BVDtype, isNotFlat*isLimited*typeval);
		}

		const int finalType = (shockType > 0)*(shockType_extend > 1)*(-2) + (BVDtype > 0);
		shockIndicator[2*celli+1] = finalType;

		const bool isShock = finalType < 0;
		const float pLimiter = limiter[celli].pLimiter;
		const float UxLimiter = limiter[celli].UxLimiter;
		const float UyLimiter = limiter[celli].UyLimiter;
		const float TLimiter = limiter[celli].TLimiter;
		limiter[celli].pLimiter = transFormLimiter(pLimiter, isShock);
		limiter[celli].UxLimiter = transFormLimiter(UxLimiter, isShock);
		limiter[celli].UyLimiter = transFormLimiter(UyLimiter, isShock);
		limiter[celli].TLimiter = transFormLimiter(TLimiter, isShock);

	}
}
#endif