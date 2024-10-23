#ifndef cudaLimiter_H
#define cudaLimiter_H

#define SMALL 1e-5f
#define VSMALL 1e-10f
#define BVD_FACTOR 0.8f
#define VALID_LIM_FACTOR 0.9f

#define ToMinmod(yy) fminf(yy, 1.0f)
#define ToBarthJespersen(yy) fminf(2.0f*yy, 1.0f)
#define ToVenkatakrishnan(yy) fminf((yy*yy + yy + VSMALL)/(yy*yy + 0.5f + 0.5f*yy + VSMALL), 1.0f)
#define ToSuperbee(yy) fmaxf(fmin(2.0f*yy, 1.0f), yy)

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
	const int cellsNum,
	const int maxStencilSize,
	const int maxCompactStencilSize,
	const int maxLocalBlockStencilSize
)
{
	const int celli = threadIdx.x + blockIdx.x*blockDim.x;

	extern __shared__ double sm_reconstruct[];
	double* x0Stencil = sm_reconstruct;
	double* y0Stencil = (double*)&x0Stencil[maxLocalBlockStencilSize];
	float* pStencil = (float*)&y0Stencil[maxLocalBlockStencilSize];
	float* UxStencil = (float*)&pStencil[maxLocalBlockStencilSize];
	float* UyStencil = (float*)&UxStencil[maxLocalBlockStencilSize];
	float* TStencil = (float*)&UyStencil[maxLocalBlockStencilSize];
	int* idStencil = (int*)&TStencil[maxLocalBlockStencilSize];
	uint16_t* stencil = (uint16_t*)&idStencil[maxLocalBlockStencilSize];
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
		idStencil[threadIdx.x] = celli;

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
			x0Stencil[blockDim.x + id] = CELL[ii].x;
			y0Stencil[blockDim.x + id] = CELL[ii].y;
			idStencil[blockDim.x + id] = ii;
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
				const int faceI = cellFaces[maxCompactStencilSize*celli + fi];
				const double dx = FACE[faceI].x - x0Stencil[threadIdx.x];
				const double dy = FACE[faceI].y - y0Stencil[threadIdx.x];
				const uint16_t& faceNeiLocalID = stencil[maxStencilSize*threadIdx.x + fi];
				const bool isNei = celli > idStencil[faceNeiLocalID];

				if (idStencil[faceNeiLocalID] >= cellsNum) continue;  // ignore boundary face direction

				faceFD[2*faceI + isNei].p = (double)(pStencil[threadIdx.x]) + tGradPx*dx + tGradPy*dy;
				faceFD[2*faceI + isNei].Ux = (double)(UxStencil[threadIdx.x]) + tGradUxx*dx + tGradUyx*dy;
				faceFD[2*faceI + isNei].Uy = (double)(UyStencil[threadIdx.x]) + tGradUxy*dx + tGradUyy*dy;
				faceFD[2*faceI + isNei].T = (double)(TStencil[threadIdx.x]) + tGradTx*dx + tGradTy*dy;
			}

			shockIndicator[celli] = 0;
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
				const int faceI = cellFaces[maxCompactStencilSize*celli + fi];
				const double dx = FACE[faceI].x - x0Stencil[threadIdx.x];
				const double dy = FACE[faceI].y - y0Stencil[threadIdx.x];
				const uint16_t& faceNeiLocalID = stencil[maxStencilSize*threadIdx.x + fi];
				const float deltaR = hypot(x0Stencil[faceNeiLocalID] - x0Stencil[threadIdx.x], y0Stencil[faceNeiLocalID] - y0Stencil[threadIdx.x]);
				const float fraction = sqrt(dx*dx + dy*dy)/deltaR;
				const bool isNei = celli > idStencil[faceNeiLocalID];

				if (idStencil[faceNeiLocalID] >= cellsNum) continue;  // ignore boundary face direction

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

			limiter[celli].pLimiter = fmax(tmpPlimiter, 0.0f)*isNotTrouble;
			limiter[celli].UxLimiter = fmax(tmpUxLimiter, 0.0f)*isNotTrouble;
			limiter[celli].UyLimiter = fmax(tmpUyLimiter, 0.0f)*isNotTrouble;
			limiter[celli].TLimiter = fmax(tmpTLimiter, 0.0f)*isNotTrouble;
			smoothnessValue = smoothnessValue/(0.5f*(pMax + pMin) + VSMALL);
			shockIndicator[celli] = (smoothnessValue > 1.0f);
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
	int* idStencil = (int*)&TLimiterStencil[blockDim.x];
	uint16_t* stencil = (uint16_t*)&idStencil[maxLocalBlockStencilSize];
	uint8_t* localStencilSize = (uint8_t*)&stencil[maxStencilSize*blockDim.x];
	int8_t* shockIndicatorStencil = (int8_t*)&localStencilSize[blockDim.x];

	if (celli < cellsNum)
	{
		localStencilSize[threadIdx.x] = stencilSize[celli];
		TStencil[threadIdx.x] = FD[celli].T;
		gradTxStencil[threadIdx.x] = gradFD[celli].gradTx;
		gradTyStencil[threadIdx.x] = gradFD[celli].gradTy;
		x0Stencil[threadIdx.x] = CELL[celli].x;
		y0Stencil[threadIdx.x] = CELL[celli].y;
		TLimiterStencil[threadIdx.x] = limiter[celli].TLimiter;
		shockIndicatorStencil[threadIdx.x] = shockIndicator[celli];
		idStencil[threadIdx.x] = celli;

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
			shockIndicatorStencil[blockDim.x + id] = shockIndicator[ii];
			idStencil[blockDim.x + id] = ii;
		}
	}

	__syncthreads();

	if (celli < cellsNum)
	{
		int BVDtype = 0;

		const int localCompactStencilSize = compactStencilSize[celli];

		for(unsigned fi = 0; fi < localCompactStencilSize; ++fi)
		{
			const int faceI = cellFaces[maxCompactStencilSize*celli + fi];
			const uint16_t& faceNeiLocalID = stencil[maxStencilSize*threadIdx.x + fi];
			const bool isNei = celli > idStencil[faceNeiLocalID];

			if (idStencil[faceNeiLocalID] >= cellsNum) continue;  // ignore boundary face direction

			const double dx_cur = FACE[faceI].x - x0Stencil[threadIdx.x];
			const double dy_cur = FACE[faceI].y - y0Stencil[threadIdx.x];

			const float TfOwn = faceFD[2*faceI].T;
			const float TfNei = faceFD[2*faceI+1].T;

			const float Tf_cur = isNei*TfNei + (1 - isNei)*TfOwn;
			const float Tf_nei = (1 - isNei)*TfNei + isNei*TfOwn;

			const float lim = ToBarthJespersen(TLimiterStencil[threadIdx.x]);
			const float Tf_cur_TVD = TStencil[threadIdx.x] + lim*(float)(gradTxStencil[threadIdx.x]*dx_cur + gradTyStencil[threadIdx.x]*dy_cur);

			const int typeval = fabsf(Tf_cur_TVD - Tf_nei) < BVD_FACTOR*fabsf(Tf_cur - Tf_nei);
			const int isNotFlat = fabsf(Tf_cur - Tf_nei) > VSMALL;
			const int isLimited = lim < VALID_LIM_FACTOR;

			BVDtype = max(BVDtype, isNotFlat*isLimited*typeval);
		}

		int8_t shockType = shockIndicatorStencil[threadIdx.x];

		for (int ci = 0; ci < localStencilSize[threadIdx.x]; ++ci)
		{
			const uint16_t& j = stencil[maxStencilSize*threadIdx.x + ci];
			shockType += shockIndicatorStencil[j];
		}

		const int finalType = (shockType > 0)*(-2) + (BVDtype > 0);
		shockIndicator[celli] = finalType;

		const bool isShock = finalType < 0;
		const float pLimiter = limiter[celli].pLimiter;
		const float UxLimiter = limiter[celli].UxLimiter;
		const float UyLimiter = limiter[celli].UyLimiter;
		const float TLimiter = limiter[celli].TLimiter;
		limiter[celli].pLimiter = isShock*(ToMinmod(pLimiter)) + (1 - isShock)*(ToBarthJespersen(pLimiter));
		limiter[celli].UxLimiter = isShock*(ToMinmod(UxLimiter)) + (1 - isShock)*(ToBarthJespersen(UxLimiter));
		limiter[celli].UyLimiter = isShock*(ToMinmod(UyLimiter)) + (1 - isShock)*(ToBarthJespersen(UyLimiter));
		limiter[celli].TLimiter = isShock*(ToMinmod(TLimiter)) + (1 - isShock)*(ToBarthJespersen(TLimiter));

	}
}

#endif