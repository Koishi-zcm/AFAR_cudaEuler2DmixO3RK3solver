#ifndef basicCUDAsolver_H
#define basicCUDAsolver_H

#include "basicDataStructure.h"
#include "fvMesh.H"
#include "fvc.H"
#include "Time.H"

class basicCUDAsolver
{
private:
	fieldPointer hostPtr_;
	fieldPointer devicePtr_;

	const Foam::fvMesh& mesh;
	Foam::volScalarField pressure;
	Foam::volVectorField velocity;
	Foam::volScalarField temperature;
	Foam::volScalarField shockIndicator;

	double R_;
	double Cv_;
	double CFL_;
	double shapeFactor_;

	int cellsNum_;
	int facesNum_;
	int patchesNum_;
	int totalBoundaryFacesNum_;

	int maxStencilSize_;
	int maxCompactStencilSize_;
	int maxLocalBlockStencilSize_;

	Foam::word stencilType_;

	void initFieldInfo();

	void initStencilData();

public:
	basicCUDAsolver(const Foam::Time& runTime, const Foam::fvMesh& mesh0);

	~basicCUDAsolver();

	void setDeltaT(Foam::Time& runTime);

	void evolve(const double deltaT);

	void writeField(const Foam::Time& runTime);
};

#endif