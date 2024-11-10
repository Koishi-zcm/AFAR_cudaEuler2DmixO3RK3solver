#ifndef basicDataStructure_H
#define basicDataStructure_H
#include <cstddef>
#include <cstdint>

#define BLOCKDIM 128
#define MAX_SHARED_MEMORY_SIZE 49152
#define MAX_VALID_STENCIL_SIZE 20

struct meshCellData
{
	double x;
	double y;
};

struct meshFaceData
{
	double x;
	double y;
	double magSf;
	double Sfx;
	double Sfy;
};

struct basicFieldData
{
    double p;
    double Ux;
    double Uy;
    double T;
};

struct conservedFieldData
{
	double rho;
	double rhoUx;
	double rhoUy;
	double rhoE;
};

struct gradientFieldData
{
	double gradPx;
	double gradPy;
	double gradUxx; double gradUxy;
	double gradUyx; double gradUyy;
	double gradTx;
	double gradTy;
};

struct basicFluxData
{
	double rhoFlux;
	double rhoUxFlux;
	double rhoUyFlux;
	double rhoEFlux;
};

struct residualFieldData
{
	double Rrho;
	double RrhoUx;
	double RrhoUy;
	double RrhoE;
};

struct limiterFieldData
{
	float pLimiter;
	float UxLimiter;
	float UyLimiter;
	float TLimiter;
};

struct fieldPointer
{
	// cell centre position. Boundary face centre positions are also recorded here
	// array size = internal cells number + boundary faces number
	meshCellData* CELL;

	// face data: position, area, face vector. Boundary face centre positions, area and face vector are also recorded here
	// array size = internal faces number + boundary faces number
	meshFaceData* FACE;

	// cell volume field
	double* cellVolume;

	// pressure, velocity, temperature field data. Boundary field data is also recorded here
	// array size = internal cells number + boundary faces number
	basicFieldData* FD;

	// TVD limiter coefficient used to suppress slope.
	limiterFieldData* limiter;

	int8_t* shockIndicator;

	// density, momentum, total energy field data
	conservedFieldData* cFD;
	// old time conserved field data of current time step, used for SSP Runge-Kutta time evolution
	conservedFieldData* cFDold;

	// gradient field data of pressure, velocity, temperature
	gradientFieldData* gradFD;

	// rhoFlux, rhoUFlux, rhoEFlux data. Boundary flux data is also recorded here
	// array size = internal faces number + boundary faces number
	basicFluxData* Flux;

	// residual of rho, rhoUx, rhoUy, rhoE
	residualFieldData* Res;

	// owner label and neighbour label of internal faces, stored as 1D array
	// neighbour[2*faceI + 0] -> own
	// neighbour[2*faceI + 1] -> nei
	int* neighbour;

	// faces number of each patch
	// boundaryFacesNum[patchi] -> patchFacesNum
	int* boundaryFacesNum;

	// label of the cell near the boundary faces, stored as 1D array
	int* boundaryFaceNeiLabel;

	// index of the boundary type for every boundary faces, stored as 1D array
	uint8_t* boundaryFacesType;

	// stencil size of each cell
	uint8_t* stencilSize;
	uint8_t* compactStencilSize;

	// RBF basis for each internal face
	// RBFbasis[2*(maxStencilSize+1)*faceI + i] -> own side basis
	// RBFbasis[2*(maxStencilSize+1)*faceI + i + (maxStencilSize+1)] -> nei side basis
	float* RBFbasis;

	// reconstructed field value on internal faces
	// faceFD[2*faceI].p -> own side reconstructed pressure value
	// faceFD[2*faceI + 1].p -> nei side reconstructed pressure value
	basicFieldData* faceFD;

	// extend stencil of each block, local cells are not included here and only additional cells are marked
	int* extendStencil;
	uint16_t* extendStencilSize;
	uint16_t* compactExtendStencilSize;

	// block local stencil label list of each cell
	uint16_t* localStencil;

	// least square inverse matrix
	float* matrix;

	// id list of faces that construct each cell, including boundary faces
	// modified face labels are stored here according to Gauss theorem direction.
	// own to nei -> cellFaces[maxCompactStencilSize*celli + fi] = (faceI + 1)
	// nei to own -> cellFaces[maxCompactStencilSize*celli + fi] = -(faceI + 1)
	int* cellFaces;

	// deltaT list for courant number evaluation in kernal function to determine time step
	double* minDeltaT;
};

#endif