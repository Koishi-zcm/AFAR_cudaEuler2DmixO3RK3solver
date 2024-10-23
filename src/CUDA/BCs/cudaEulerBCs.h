#ifndef cudaEulerBCs_H
#define cudaEulerBCs_H

#define FUNCLIST(name) void name( \
	double& bp, double& bUx, double& bUy, double& bT, \
	const double& pLeft, \
	const double& UxLeft, \
	const double& UyLeft, \
	const double& TLeft, \
	const double& R, const double& Cv, \
	const double& cx, const double& cy, \
	const double& bx, const double& by, \
	const double& magSf, const double& Sfx, const double& Sfy \
)

__device__ FUNCLIST(constantTemperatureWallBoundary)
{
	bp = pLeft;
	bUx = 0.0;
	bUy = 0.0;
	bT = 1.0;
}

__device__ FUNCLIST(adiabaticWallBoundary)
{
	bp = pLeft;
	bUx = 0.0;
	bUy = 0.0;
	bT = TLeft;
}

__device__ FUNCLIST(symmetryBoundary)
{
	const double nfx = Sfx/magSf;
	const double nfy = Sfy/magSf;
	bp = pLeft;
	bUx = UxLeft - (UxLeft*nfx + UyLeft*nfy)*nfx;
	bUy = UyLeft - (UxLeft*nfx + UyLeft*nfy)*nfy;
	bT = TLeft;
}

__device__ FUNCLIST(fixedValueBoundary)
{
	bp = 1.0;
	bUx = 3.0;
	bUy = 0.0;
	bT = 1.0;
}

__device__ FUNCLIST(outletBoundary)
{
	bp = pLeft;
	bUx = UxLeft;
	bUy = UyLeft;
	bT = TLeft;
}

__device__ FUNCLIST(fixedValue2Boundary)
{
	bp = 30.05945;
	bUx = 4.07795;
	bUy = 0.0;
	bT = 5.97677;
}


__device__ void (*updateBoundaryValue[])(
	double& bp, double& bUx, double& bUy, double& bT,
	const double& pLeft,
	const double& UxLeft,
	const double& UyLeft,
	const double& TLeft,
	const double& R, const double& Cv,
	const double& cx, const double& cy,
	const double& bx, const double& by,
	const double& magSf, const double& Sfx, const double& Sfy
) = {
	constantTemperatureWallBoundary,  // 0
	adiabaticWallBoundary,  // 1
	symmetryBoundary,  // 2
	fixedValueBoundary,  // 3
	outletBoundary,  // 4
	fixedValue2Boundary  // 5
};

__global__ void updateBoundaryFieldData(
	basicFieldData* FD,
	const meshCellData* CELL,
	const meshFaceData* FACE,
	const int* boundaryFaceNeiLabel,
	const uint8_t* boundaryFacesType,
	const int cellsNum,
	const int facesNum,
	const int totalBoundaryFacesNum,
	const double R,
	const double Cv
)
{
	const int tfacei = threadIdx.x + blockIdx.x*blockDim.x;

	if (tfacei < totalBoundaryFacesNum)
	{
		const int& curFC = boundaryFaceNeiLabel[tfacei];
		const uint8_t& type = boundaryFacesType[tfacei];
		updateBoundaryValue[type](
			FD[tfacei + cellsNum].p,
			FD[tfacei + cellsNum].Ux,
			FD[tfacei + cellsNum].Uy,
			FD[tfacei + cellsNum].T,
			FD[curFC].p, FD[curFC].Ux, FD[curFC].Uy, FD[curFC].T, 
			R, Cv,
			CELL[curFC].x, CELL[curFC].y,
			FACE[tfacei + facesNum].x, FACE[tfacei + facesNum].y,
			FACE[tfacei + facesNum].magSf, FACE[tfacei + facesNum].Sfx, FACE[tfacei + facesNum].Sfy
		);
	}
}




#endif