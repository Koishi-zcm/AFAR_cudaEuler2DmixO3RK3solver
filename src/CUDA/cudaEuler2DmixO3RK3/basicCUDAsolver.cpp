#include "basicCUDAsolver.h"
#include "GPUevolve.h"
#include "fvc.H"
#include "d2GaussQuadrature.H"
#include <set>
#include <iostream>

basicCUDAsolver::basicCUDAsolver(const Foam::Time& runTime, const Foam::fvMesh& mesh0)
:
    hostPtr_(),
    devicePtr_(),
    mesh(mesh0),
    pressure(
        Foam::IOobject(
            "p",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh
    ),
    velocity(
        Foam::IOobject(
            "U",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh
    ),
    temperature(
        Foam::IOobject(
            "T",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh
    ),
    shockIndicator(
        Foam::IOobject(
            "shockIndicator",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar(Foam::dimensionSet(0,0,0,0,0,0,0), 0.0)
    ),
    R_(Foam::readScalar(mesh.schemes().subDict("CUDA").lookup("R"))),
    Cv_(Foam::readScalar(mesh.schemes().subDict("CUDA").lookup("Cv"))),
    CFL_(Foam::readScalar(runTime.controlDict().lookup("maxCo"))),
    shapeFactor_(mesh.schemes().lookupOrDefault("shapeFactor", 10.0)),
    cellsNum_(mesh.nCells()),
    facesNum_(mesh.nInternalFaces()),
    patchesNum_(0),
    totalBoundaryFacesNum_(0),
    maxStencilSize_(0),
    maxCompactStencilSize_(0),
    maxLocalBlockStencilSize_(0),
    stencilType_(mesh.schemes().subDict("CUDA").lookup("stencilType"))
{
    Foam::Info << "initializing..." << Foam::endl;
    
    if (mesh.nSolutionD() != 2)
    {
        Foam::Info << "This solver is only used for 2D mesh case! Abort..." << Foam::endl;
        std::exit(-1);
    }

    initFieldInfo();
    initStencilData();

    Foam::Info << "initializing device field data..." << Foam::endl;
    setDeviceFieldData(
        devicePtr_, hostPtr_,
        cellsNum_, facesNum_,
        patchesNum_, totalBoundaryFacesNum_,
        maxStencilSize_, maxCompactStencilSize_, maxLocalBlockStencilSize_
    );
}


basicCUDAsolver::~basicCUDAsolver()
{
    freeFieldData(devicePtr_, hostPtr_);
    std::cout << "delete device field data successfully" << std::endl;

    delete[] hostPtr_.FACE;
    delete[] hostPtr_.Flux;
    delete[] hostPtr_.neighbour;
    delete[] hostPtr_.CELL;
    delete[] hostPtr_.cellVolume;
    delete[] hostPtr_.FD;
    delete[] hostPtr_.cFD;
    delete[] hostPtr_.gradFD;
    delete[] hostPtr_.Res;
    delete[] hostPtr_.stencilSize;
    delete[] hostPtr_.cellFaces;
    delete[] hostPtr_.boundaryFacesNum;
    delete[] hostPtr_.boundaryFaceNeiLabel;
    delete[] hostPtr_.boundaryFacesType;
    delete[] hostPtr_.extendStencilSize;
    delete[] hostPtr_.localStencil;
    delete[] hostPtr_.extendStencil;
    delete[] hostPtr_.minDeltaT;
    std::cout << "delete host field data successfully" << std::endl;
}


void basicCUDAsolver::initFieldInfo()
{
    const Foam::surfaceVectorField& faceCentre = mesh.Cf();
    const Foam::surfaceVectorField& Sf = mesh.Sf();
    const Foam::surfaceScalarField& magSf = mesh.magSf();

    for (int patchi = 0; patchi < faceCentre.boundaryField().size(); ++patchi)
    {
        const Foam::fvsPatchVectorField& pFaceCentre = faceCentre.boundaryField()[patchi];
        if (pFaceCentre.size() != 0)
        {
            patchesNum_++;
            totalBoundaryFacesNum_ += pFaceCentre.size();
        }
    }
    Foam::Info << "patches number: " << patchesNum_ << Foam::endl;
    Foam::Info << "total boundary faces number: " << totalBoundaryFacesNum_ << Foam::endl;

    hostPtr_.boundaryFacesNum = new int[patchesNum_];
    hostPtr_.boundaryFaceNeiLabel = new int[totalBoundaryFacesNum_];
    hostPtr_.boundaryFacesType = new uint8_t[totalBoundaryFacesNum_];

    Foam::scalarField boundaryFaceCentre_x(totalBoundaryFacesNum_, 0.0);
    Foam::scalarField boundaryFaceCentre_y(totalBoundaryFacesNum_, 0.0);
    Foam::scalarField boundaryFaceArea(totalBoundaryFacesNum_, 0.0);
    Foam::scalarField boundaryFaceVector_x(totalBoundaryFacesNum_, 0.0);
    Foam::scalarField boundaryFaceVector_y(totalBoundaryFacesNum_, 0.0);
    Foam::scalarField boundaryPressure(totalBoundaryFacesNum_, 0.0);
    Foam::scalarField boundaryVelocity_x(totalBoundaryFacesNum_, 0.0);
    Foam::scalarField boundaryVelocity_y(totalBoundaryFacesNum_, 0.0);
    Foam::scalarField boundaryTemperature(totalBoundaryFacesNum_, 0.0);

    int idx = 0;
    int tfacei = 0;
    for (int patchi = 0; patchi < faceCentre.boundaryField().size(); ++patchi)
    {
        const Foam::fvsPatchVectorField& pFaceCentre = faceCentre.boundaryField()[patchi];
        const Foam::fvsPatchVectorField& pSf = Sf.boundaryField()[patchi];
        const Foam::fvsPatchScalarField& pMagSf = magSf.boundaryField()[patchi];
        const Foam::fvPatchScalarField& pp = pressure.boundaryField()[patchi];
        const Foam::fvPatchVectorField& pU = velocity.boundaryField()[patchi];
        const Foam::fvPatchScalarField& pT = temperature.boundaryField()[patchi];
        const Foam::labelList& fc = pFaceCentre.patch().faceCells();

        if (pFaceCentre.size() == 0) continue;

        Foam::Info << "patchi = " << patchi << ", boundaryPatchName: " << pFaceCentre.patch().name();

        uint8_t typeLabel = 4;
        Foam::word patchIDname = "patch" + Foam::word(std::to_string(patchi));
        typeLabel = Foam::readLabel(mesh.schemes().subDict("CUDA").subDict("boundary").lookup(patchIDname));

        Foam::Info << ", typeLabel = " << typeLabel << Foam::endl;

        hostPtr_.boundaryFacesNum[idx] = pFaceCentre.size();
        idx++;

        for (int i = 0; i < pFaceCentre.size(); ++i)
        {
            hostPtr_.boundaryFaceNeiLabel[tfacei] = fc[i];
            hostPtr_.boundaryFacesType[tfacei] = typeLabel;

            boundaryFaceCentre_x[tfacei] = pFaceCentre[i].x();
            boundaryFaceCentre_y[tfacei] = pFaceCentre[i].y();
            boundaryFaceArea[tfacei] = pMagSf[i];
            boundaryFaceVector_x[tfacei] = pSf[i].x();
            boundaryFaceVector_y[tfacei] = pSf[i].y();
            boundaryPressure[tfacei] = pp[i];
            boundaryVelocity_x[tfacei] = pU[i].x();
            boundaryVelocity_y[tfacei] = pU[i].y();
            boundaryTemperature[tfacei] = pT[i];

            tfacei++;
        }
    }

    Foam::Info << "initializing cell field data..." << Foam::endl;

    const Foam::volVectorField& cellCentre = mesh.C();
    const Foam::scalarField& cellV = mesh.V();

    hostPtr_.CELL = new meshCellData[cellsNum_ + totalBoundaryFacesNum_];
    hostPtr_.cellVolume = new double[cellsNum_];
    hostPtr_.FD = new basicFieldData[cellsNum_ + totalBoundaryFacesNum_];
    hostPtr_.cFD = new conservedFieldData[cellsNum_];
    hostPtr_.gradFD = new gradientFieldData[cellsNum_];
    hostPtr_.shockIndicator = new int8_t[cellsNum_ + totalBoundaryFacesNum_];
    hostPtr_.Res = new residualFieldData[cellsNum_];

    for (int i = 0; i < cellsNum_; ++i)
    {
        hostPtr_.CELL[i].x = cellCentre[i].x();
        hostPtr_.CELL[i].y = cellCentre[i].y();

        hostPtr_.cellVolume[i] = cellV[i];

        hostPtr_.FD[i].p = pressure[i];
        hostPtr_.FD[i].Ux = velocity[i].x();
        hostPtr_.FD[i].Uy = velocity[i].y();
        hostPtr_.FD[i].T = temperature[i];

        hostPtr_.cFD[i].rho = pressure[i]/(R_*temperature[i]);
        hostPtr_.cFD[i].rhoUx = hostPtr_.cFD[i].rho*velocity[i].x();
        hostPtr_.cFD[i].rhoUy = hostPtr_.cFD[i].rho*velocity[i].y();
        hostPtr_.cFD[i].rhoE = hostPtr_.cFD[i].rho*(Cv_*temperature[i] + 0.5*(velocity[i].x()*velocity[i].x() + velocity[i].y()*velocity[i].y()));

        hostPtr_.gradFD[i].gradPx = 0.0;  hostPtr_.gradFD[i].gradPy = 0.0;
        hostPtr_.gradFD[i].gradUxx = 0.0; hostPtr_.gradFD[i].gradUxy = 0.0;
        hostPtr_.gradFD[i].gradUyx = 0.0; hostPtr_.gradFD[i].gradUyy = 0.0;
        hostPtr_.gradFD[i].gradTx = 0.0;  hostPtr_.gradFD[i].gradTy = 0.0;

        hostPtr_.shockIndicator[i] = 0;

        hostPtr_.Res[i].Rrho = 0.0;
        hostPtr_.Res[i].RrhoUx = 0.0;
        hostPtr_.Res[i].RrhoUy = 0.0;
        hostPtr_.Res[i].RrhoE = 0.0;
    }

    for (int i = cellsNum_; i < cellsNum_ + totalBoundaryFacesNum_; ++i)
    {
        hostPtr_.CELL[i].x = boundaryFaceCentre_x[i - cellsNum_];
        hostPtr_.CELL[i].y = boundaryFaceCentre_y[i - cellsNum_];

        hostPtr_.FD[i].p = boundaryPressure[i - cellsNum_];
        hostPtr_.FD[i].Ux = boundaryVelocity_x[i - cellsNum_];
        hostPtr_.FD[i].Uy = boundaryVelocity_y[i - cellsNum_];
        hostPtr_.FD[i].T = boundaryTemperature[i - cellsNum_];

        hostPtr_.shockIndicator[i] = 0;
    }

    Foam::Info << "initializing face field data..." << Foam::endl;

    hostPtr_.FACE = new meshFaceData[facesNum_ + totalBoundaryFacesNum_];
    hostPtr_.Flux = new basicFluxData[facesNum_ + totalBoundaryFacesNum_];

    for (int i = 0; i < facesNum_; ++i)
    {
        hostPtr_.FACE[i].x = faceCentre[i].x();
        hostPtr_.FACE[i].y = faceCentre[i].y();
        hostPtr_.FACE[i].magSf = magSf[i];
        hostPtr_.FACE[i].Sfx = Sf[i].x();
        hostPtr_.FACE[i].Sfy = Sf[i].y();

        hostPtr_.Flux[i].rhoFlux = 0.0;
        hostPtr_.Flux[i].rhoUxFlux = 0.0;
        hostPtr_.Flux[i].rhoUyFlux = 0.0;
        hostPtr_.Flux[i].rhoEFlux = 0.0;
    }

    for (int i = facesNum_; i < facesNum_ + totalBoundaryFacesNum_; ++i)
    {
        hostPtr_.FACE[i].x = boundaryFaceCentre_x[i - facesNum_];
        hostPtr_.FACE[i].y = boundaryFaceCentre_y[i - facesNum_];
        hostPtr_.FACE[i].magSf = boundaryFaceArea[i - facesNum_];
        hostPtr_.FACE[i].Sfx = boundaryFaceVector_x[i - facesNum_];
        hostPtr_.FACE[i].Sfy = boundaryFaceVector_y[i - facesNum_];

        hostPtr_.Flux[i].rhoFlux = 0.0;
        hostPtr_.Flux[i].rhoUxFlux = 0.0;
        hostPtr_.Flux[i].rhoUyFlux = 0.0;
        hostPtr_.Flux[i].rhoEFlux = 0.0;
    }
}


void basicCUDAsolver::initStencilData()
{
    int* globalStencil = nullptr;
    int* globalExtendCells = nullptr;
    const int maxValidStencilSize = MAX_VALID_STENCIL_SIZE;
    const int maxSharedMemSize = MAX_SHARED_MEMORY_SIZE;
    const int blockDim = BLOCKDIM;

    Foam::Info << "initializing neighbour information..." << Foam::endl;

    const Foam::unallocLabelList& owner_orig = mesh.owner();
    const Foam::unallocLabelList& neighbour_orig = mesh.neighbour();
    
    hostPtr_.neighbour = new int[facesNum_*2];
    Foam::List<Foam::labelList> cellFaces_orig(mesh.nCells());
    Foam::List<Foam::List<int>> faceDirection_orig(mesh.nCells());
    for (int faceI = 0; faceI < facesNum_; ++faceI)
    {
        hostPtr_.neighbour[2*faceI] = owner_orig[faceI];
        hostPtr_.neighbour[2*faceI+1] = neighbour_orig[faceI];

        cellFaces_orig[owner_orig[faceI]].append(faceI);
        cellFaces_orig[neighbour_orig[faceI]].append(faceI);

        const int owndir = 1;
        const int neidir = -1;
        faceDirection_orig[owner_orig[faceI]].append(owndir);
        faceDirection_orig[neighbour_orig[faceI]].append(neidir);
    }

    for (int i = 0; i < totalBoundaryFacesNum_; ++i)
    {
        const int celli = hostPtr_.boundaryFaceNeiLabel[i];
        const int faceI = i + facesNum_;
        const int owndir = 1;
        cellFaces_orig[celli].append(faceI);
        faceDirection_orig[celli].append(owndir);
    }


    Foam::Info << "initializing cellCells information..." << Foam::endl;

    const Foam::labelListList& cellCells_orig = mesh.cellCells();
    Foam::List<Foam::labelList> cellCells(cellsNum_);
    for (int i = 0; i < cellsNum_; ++i) cellCells[i] = cellCells_orig[i];
    for (int i = 0; i < totalBoundaryFacesNum_; ++i)
    {
        const int celli = hostPtr_.boundaryFaceNeiLabel[i];
        const int bfid = cellsNum_ + i;
        cellCells[celli].append(bfid);
    }

    Foam::Info << "initializing stencil information..." << Foam::endl;

    const Foam::labelListList& cellEdges = mesh.cellEdges();
    const Foam::labelListList& edgeCells = mesh.edgeCells();
    hostPtr_.stencilSize = new uint8_t[cellsNum_];
    hostPtr_.compactStencilSize = new uint8_t[cellsNum_];
    int* globalStencil_orig = new int[maxValidStencilSize*cellsNum_];
    for (int celli = 0; celli < cellsNum_; ++celli)
    {
        if (celli % 10000 == 0 || celli == cellsNum_-1)
        {
            std::cout << std::unitbuf 
                << "collecting basic stencil information...[" 
                << std::to_string(celli+1) << '/' 
                << std::to_string(cellsNum_) << ']' << '\r';
        }

        Foam::labelList tmpStencilAddr(maxValidStencilSize, 0);
        std::set<Foam::label> addedAddr;

        addedAddr.insert(celli);

        int idx = 0;

        const Foam::labelList& cc = cellCells[celli];
        forAll(cc, i)
        {
            tmpStencilAddr[idx] = cc[i];
            idx++;
            addedAddr.insert(cc[i]);
        }

        if (stencilType_ == "faceNeighbour")  // second layer face-neighbouring cells
        {
            forAll(cc, i)
            {
                const Foam::label& faceNeiID = cc[i];

                if (faceNeiID >= cellsNum_) continue;

                const Foam::labelList& faceNeiNei = cellCells[faceNeiID];

                forAll(faceNeiNei, j)
                {
                    const Foam::label& faceNeiNeiID = faceNeiNei[j];
                    if (addedAddr.find(faceNeiNeiID) == addedAddr.end())
                    {
                        // tmpStencilAddr.append(faceNeiNeiID);
                        tmpStencilAddr[idx] = faceNeiNeiID;
                        idx++;
                        addedAddr.insert(faceNeiNeiID);
                    }
                }
            }
        }
        else if (stencilType_ == "edgeNeighbour")  // edge neighbouring cells
        {
            const Foam::labelList& edgesList = cellEdges[celli];
            forAll(edgesList, i)
            {
                const Foam::label& edgeID = edgesList[i];
                const Foam::labelList& edgeNeiList = edgeCells[edgeID];

                forAll(edgeNeiList, j)
                {
                    const Foam::label& edgeNeiID = edgeNeiList[j];
                    if (addedAddr.find(edgeNeiID) == addedAddr.end())
                    {
                        tmpStencilAddr[idx] = edgeNeiID;
                        idx++;
                        addedAddr.insert(edgeNeiID);
                    }
                }
            }
        }
        else
        {
            Foam::Info << "Error: Invalid stencil type setting. Valid settings are: faceNeighbour, edgeNeighbour" << Foam::endl;
            std::exit(-1);
        }

        for (int i = 0; i < idx; ++i)
        {
            globalStencil_orig[maxValidStencilSize*celli + i] = tmpStencilAddr[i];
        }

        hostPtr_.stencilSize[celli] = idx;
        hostPtr_.compactStencilSize[celli] = cc.size();

        maxStencilSize_ = Foam::max(maxStencilSize_, idx);
        maxCompactStencilSize_ = Foam::max(maxCompactStencilSize_, cc.size());

        if (maxStencilSize_ > maxValidStencilSize)
        {
            Foam::Info << "Error: Too large stencil! maxStencilSize = " << maxStencilSize_ << Foam::endl;
            std::exit(-1);
        }
    }
    std::cout << std::endl;
    Foam::Info << "maxStencilSize: " << maxStencilSize_ << Foam::endl;


    hostPtr_.cellFaces = new int[maxCompactStencilSize_*cellsNum_];
    hostPtr_.faceDirection = new int8_t[maxCompactStencilSize_*cellsNum_];
    globalStencil = new int[maxStencilSize_*cellsNum_];
    for (int celli = 0; celli < cellsNum_; ++celli)
    {
        for (int i = 0; i < hostPtr_.compactStencilSize[celli]; ++i)
        {
            hostPtr_.cellFaces[maxCompactStencilSize_*celli + i] = cellFaces_orig[celli][i];
            hostPtr_.faceDirection[maxCompactStencilSize_*celli + i] = faceDirection_orig[celli][i];
        }

        for (int i = 0; i < hostPtr_.stencilSize[celli]; ++i)
        {
            globalStencil[maxStencilSize_*celli + i] = globalStencil_orig[maxValidStencilSize*celli + i];
        }
    }
    delete[] globalStencil_orig;

    Foam::Info << "collecting gauss quadrature information..." << Foam::endl;
    Foam::autoPtr<Foam::d2GaussQuadrature> GQ(new Foam::d2GaussQuadrature(mesh));
    const Foam::List<Foam::List<Foam::vector>>& cellGaussPoints_orig = GQ->cellGaussPoints();
    const Foam::List<Foam::List<Foam::scalar>>& cellGaussWeights_orig = GQ->cellGaussWeights();
    const Foam::List<Foam::List<Foam::List<Foam::vector>>>& boundaryFaceGaussPointsList = GQ->boundaryFaceGaussPointsList();
    const Foam::List<Foam::List<Foam::List<Foam::scalar>>>& boundaryFaceGaussWeightsList = GQ->boundaryFaceGaussWeightsList();

    Foam::List<Foam::List<Foam::vector>> cellGaussPoints(cellsNum_ + totalBoundaryFacesNum_);
    Foam::List<Foam::List<Foam::scalar>> cellGaussWeights(cellsNum_ + totalBoundaryFacesNum_);

    for (int i = 0; i < cellsNum_; ++i)
    {
        cellGaussPoints[i] = cellGaussPoints_orig[i];
        cellGaussWeights[i] = cellGaussWeights_orig[i];
    }

    int tfacei = 0;
    for (int patchi = 0; patchi < mesh.Cf().boundaryField().size(); ++patchi)
    {
        const int patchSize = mesh.Cf().boundaryField()[patchi].size();
        if (patchSize == 0) continue;

        for (int i = 0; i < patchSize; ++i)
        {
            cellGaussPoints[tfacei + cellsNum_] = boundaryFaceGaussPointsList[patchi][i];

            const Foam::List<Foam::scalar>& tmpWeights_orig = boundaryFaceGaussWeightsList[patchi][i];
            const int weightsNum = tmpWeights_orig.size();
            Foam::List<Foam::scalar> tmpWeights(weightsNum, 0.0);
            // The boundary face Gauss weights are collected for flux evaluation. So face area should be divided here for average value
            for (int k = 0; k < weightsNum; ++k) tmpWeights[k] = tmpWeights_orig[k]/mesh.magSf().boundaryField()[patchi][i];
            cellGaussWeights[tfacei + cellsNum_] = tmpWeights;

            tfacei++;
        }
    }

    Foam::scalarField refDistance(mesh.nCells(), Foam::Zero);  // reference distance for non-uniform mesh
    Foam::List<Foam::scalarRectangularMatrix> transformMatrixList(mesh.nCells());
    Foam::Field<bool> troubleMatrix(mesh.nCells(), false);
    bool existTroubleMatrix = false;
    for (int celli = 0; celli < cellsNum_; ++celli)
    {
        if (celli % 10000 == 0 || celli == cellsNum_-1)
        {
            std::cout << std::unitbuf 
                << "evaluating transformMatrix...[" 
                << std::to_string(celli+1) << '/' 
                << std::to_string(cellsNum_) << ']' << '\r';
        }

        const Foam::label& stencilSize = hostPtr_.stencilSize[celli];
        Foam::scalarRectangularMatrix P = Foam::scalarRectangularMatrix(stencilSize+1, stencilSize+1, Foam::Zero);

        for (int i = 0; i < stencilSize; ++i)
        {
            const int& sid = globalStencil[maxStencilSize_*celli + i];
            const double refdx = hostPtr_.CELL[sid].x - hostPtr_.CELL[celli].x;
            const double refdy = hostPtr_.CELL[sid].y - hostPtr_.CELL[celli].y;
            refDistance[celli] += Foam::sqrt(refdx*refdx + refdy*refdy);
        }
        refDistance[celli] /= double(stencilSize);

        for (int i = 0; i < stencilSize+1; ++i)
        {
            const int si = (i==0) ? celli : globalStencil[maxStencilSize_*celli + i-1];

            for (int j = 0; j < stencilSize+1; ++j)
            {
                const int sj = (j==0) ? celli : globalStencil[maxStencilSize_*celli + j-1];
                const double x0 = hostPtr_.CELL[sj].x;
                const double y0 = hostPtr_.CELL[sj].y;

                for (int k = 0; k < cellGaussPoints[si].size(); ++k)
                {
                    const double x = cellGaussPoints[si][k].x();
                    const double y = cellGaussPoints[si][k].y();
                    const double w = cellGaussWeights[si][k];
                    const double r2 = Foam::sqr((x - x0)/refDistance[celli]) + Foam::sqr((y - y0)/refDistance[celli]);
                    P(i,j) += w*Foam::sqrt(this->shapeFactor_ + r2);
                }
            }
        }

        transformMatrixList[celli] = Foam::SVDinv(P);
        for (int i = 0; i < stencilSize+1; ++i)
        {
            for (int j = 0; j < stencilSize+1; ++j)
            {
                const Foam::scalarRectangularMatrix& invP = transformMatrixList[celli];
                if (Foam::mag(invP(i,j)) > 1e10)
                {
                    troubleMatrix[celli] = true;
                    existTroubleMatrix = true;
                }
            }
        }
    }
    std::cout << std::endl;

    if (existTroubleMatrix) Foam::Info << "Warning: Singular transformMatrix exist!" << Foam::endl;

    const Foam::surfaceVectorField& faceCentre = mesh.Cf();
    hostPtr_.RBFbasis = new float[2*(maxStencilSize_+1)*facesNum_];
    for (int faceI = 0; faceI < facesNum_; ++faceI)
    {
        if (faceI % 10000 == 0 || faceI == facesNum_-1)
        {
            std::cout << std::unitbuf 
                << "evaluating RBF basis...[" 
                << std::to_string(faceI+1) << '/' 
                << std::to_string(facesNum_) << ']' << '\r';
        }

        const int& own = hostPtr_.neighbour[2*faceI];
        const int& nei = hostPtr_.neighbour[2*faceI+1];

        // here the stencil size has included current cell
        const int stencilSizeOwn = hostPtr_.stencilSize[own] + 1;
        const int stencilSizeNei = hostPtr_.stencilSize[nei] + 1;

        const double& refDistanceOwn = refDistance[own];
        const double& refDistanceNei = refDistance[nei];

        Foam::scalarField basisOwn(stencilSizeOwn, Foam::Zero);
        Foam::scalarField basisNei(stencilSizeNei, Foam::Zero);

        const Foam::scalar& x = faceCentre[faceI].x();
        const Foam::scalar& y = faceCentre[faceI].y();

        for (int i = 0; i < stencilSizeOwn; ++i)
        {
            const int si = (i==0) ? own : globalStencil[maxStencilSize_*own + i-1];
            const double x0 = hostPtr_.CELL[si].x;
            const double y0 = hostPtr_.CELL[si].y;
            const double r2 = Foam::sqr((x - x0)/refDistanceOwn) + Foam::sqr((y - y0)/refDistanceOwn);
            basisOwn[i] = Foam::sqrt(this->shapeFactor_ + r2);
        }

        for (int i = 0; i < stencilSizeNei; ++i)
        {
            const int si = (i==0) ? nei : globalStencil[maxStencilSize_*nei + i-1];
            const double x0 = hostPtr_.CELL[si].x;
            const double y0 = hostPtr_.CELL[si].y;
            const double r2 = Foam::sqr((x - x0)/refDistanceNei) + Foam::sqr((y - y0)/refDistanceNei);
            basisNei[i] = Foam::sqrt(this->shapeFactor_ + r2);
        }

        if (troubleMatrix[own])
        {
            hostPtr_.RBFbasis[2*(maxStencilSize_+1)*faceI] = 1.0;
            for (int i = 1; i < stencilSizeOwn; ++i) hostPtr_.RBFbasis[2*(maxStencilSize_+1)*faceI + i] = 0.0;
        }
        else
        {
            const Foam::scalarRectangularMatrix& Gown = transformMatrixList[own];
            Foam::scalarRectangularMatrix UMatOwn(stencilSizeOwn, stencilSizeOwn, 1.0);
            Foam::scalarField RBFbasisOwn = Gown.T()*basisOwn - UMatOwn*Gown.T()*basisOwn/Foam::scalar(stencilSizeOwn) + Foam::scalarField(stencilSizeOwn, 1.0)/Foam::scalar(stencilSizeOwn);
            for (int i = 0; i < stencilSizeOwn; ++i) hostPtr_.RBFbasis[2*(maxStencilSize_+1)*faceI + i] = RBFbasisOwn[i];
        }

        if (troubleMatrix[nei])
        {
            hostPtr_.RBFbasis[2*(maxStencilSize_+1)*faceI + (maxStencilSize_+1)] = 1.0;
            for (int i = 1; i < stencilSizeNei; ++i) hostPtr_.RBFbasis[2*(maxStencilSize_+1)*faceI + i + (maxStencilSize_+1)] = 0.0;
        }
        else
        {
            const Foam::scalarRectangularMatrix& Gnei = transformMatrixList[nei];
            Foam::scalarRectangularMatrix UMatNei(stencilSizeNei, stencilSizeNei, 1.0);
            Foam::scalarField RBFbasisNei = Gnei.T()*basisNei - UMatNei*Gnei.T()*basisNei/Foam::scalar(stencilSizeNei) + Foam::scalarField(stencilSizeNei, 1.0)/Foam::scalar(stencilSizeNei);
            for (int i = 0; i < stencilSizeNei; ++i) hostPtr_.RBFbasis[2*(maxStencilSize_+1)*faceI + i + (maxStencilSize_+1)] = RBFbasisNei[i];
        }
    }
    std::cout << std::endl;

    int gridDim = (cellsNum_ + blockDim - 1)/blockDim;

    hostPtr_.extendStencilSize = new uint16_t[gridDim];
    int maxValidLocalBlockStencilSize = (maxSharedMemSize - sizeof(uint8_t)*blockDim*2 - sizeof(uint16_t)*blockDim*maxStencilSize_)/(sizeof(double)*2 + sizeof(float)*4 + sizeof(int));
    maxValidLocalBlockStencilSize = Foam::min(int((maxSharedMemSize - sizeof(double)*blockDim*4 - sizeof(float)*blockDim*2 - sizeof(uint16_t)*blockDim*maxStencilSize_ - sizeof(uint8_t)*blockDim)/(sizeof(int8_t) + sizeof(int))), maxValidLocalBlockStencilSize);
    globalExtendCells = new int[maxValidLocalBlockStencilSize*gridDim];
    for (int blockIdx = 0; blockIdx < gridDim; ++blockIdx)
    {
        std::set<int> extendCellsSet;
        Foam::List<int> extendCells;

        // find extend cells and store their global label
        for (int threadIdx = 0; threadIdx < blockDim; ++threadIdx)
        {
            const int celli = threadIdx + blockIdx*blockDim;
            if (celli % 10000 == 0 || celli == cellsNum_-1)
            {
                std::cout << std::unitbuf 
                    << "changing globalStencil into localStencil data (step 1)...[" 
                    << std::to_string(celli+1) << '/' 
                    << std::to_string(cellsNum_) << ']' << '\r';
            }
            if (celli >= cellsNum_) break;

            for (int i = 0; i < hostPtr_.stencilSize[celli]; ++i)
            {
                const int extendID = globalStencil[maxStencilSize_*celli + i];
                bool isInExtendStencil = extendID < blockIdx*blockDim || extendID > blockDim-1 + blockIdx*blockDim || extendID >= cellsNum_;
                if (isInExtendStencil)
                {
                    if (extendCellsSet.find(extendID) == extendCellsSet.end())
                    {
                        extendCellsSet.insert(extendID);
                        extendCells.append(extendID);
                    }
                }
            }
        }

        const int size = extendCells.size();
        hostPtr_.extendStencilSize[blockIdx] = size;
        maxLocalBlockStencilSize_ = Foam::max(size + blockDim, maxLocalBlockStencilSize_);
        
        if (maxLocalBlockStencilSize_ > maxValidLocalBlockStencilSize)
        {
            Foam::Info << "Error: Too large local block stencil size. maxLocalBlockStencilSize = " << maxLocalBlockStencilSize_ << Foam::endl;
            std::exit(-1);
        }

        for (int i = 0; i < size; ++i)
        {
            globalExtendCells[maxValidLocalBlockStencilSize*blockIdx + i] = extendCells[i];
        }
    }
    std::cout << std::endl;
    Foam::Info << "maxLocalBlockStencilSize: " << maxLocalBlockStencilSize_ << Foam::endl;
    hostPtr_.localStencil = new uint16_t[maxStencilSize_*cellsNum_];
    hostPtr_.extendStencil = new int[(maxLocalBlockStencilSize_ - blockDim)*gridDim];

    for (int blockIdx = 0; blockIdx < gridDim; ++blockIdx)
    {
        Foam::List<int> extendCells(maxLocalBlockStencilSize_ - blockDim, 0);

        for (int i = 0; i < hostPtr_.extendStencilSize[blockIdx]; ++i)
        {
            extendCells[i] = globalExtendCells[maxValidLocalBlockStencilSize*blockIdx + i];
            hostPtr_.extendStencil[(maxLocalBlockStencilSize_ - blockDim)*blockIdx + i] = extendCells[i];
        }

        // change global stencil label into local block stencil label
        for (int threadIdx = 0; threadIdx < blockDim; ++threadIdx)
        {
            const int celli = threadIdx + blockIdx*blockDim;
            if (celli % 10000 == 0 || celli == cellsNum_-1)
            {
                std::cout << std::unitbuf 
                    << "changing globalStencil into localStencil data (step 2)...[" 
                    << std::to_string(celli+1) << '/' 
                    << std::to_string(cellsNum_) << ']' << '\r';
            }
            if (celli >= cellsNum_) break;

            for (int i = 0; i < hostPtr_.stencilSize[celli]; ++i)
            {
                const int sid = globalStencil[maxStencilSize_*celli + i];
                bool isInExtendStencil = sid < blockIdx*blockDim || sid > blockDim-1 + blockIdx*blockDim || sid >= cellsNum_;
                if (isInExtendStencil)
                {
                    int exid = 0;
                    while (extendCells[exid] != sid) exid++;
                    const int newsid = blockDim + exid;
                    hostPtr_.localStencil[maxStencilSize_*celli + i] = newsid;
                }
                else
                {
                    const int newsid = sid - blockIdx*blockDim;
                    hostPtr_.localStencil[maxStencilSize_*celli + i] = newsid;
                }
            }
        }
    }
    std::cout << std::endl;

    delete[] globalStencil;
    delete[] globalExtendCells;

    size_t sharedMemSize1 = sizeof(double)*maxLocalBlockStencilSize_*2
        + sizeof(float)*maxLocalBlockStencilSize_*4
        + sizeof(int)*maxLocalBlockStencilSize_
        + sizeof(uint16_t)*blockDim*maxStencilSize_
        + sizeof(uint8_t)*blockDim*2;
    Foam::Info << "reconstruct sharedMemSize = " << sharedMemSize1 << Foam::endl;
    size_t sharedMemSize2 = sizeof(double)*blockDim*4
        + sizeof(float)*blockDim*2
        + sizeof(int)*maxLocalBlockStencilSize_
        + sizeof(uint16_t)*blockDim*maxStencilSize_
        + sizeof(uint8_t)*blockDim
        + sizeof(int8_t)*maxLocalBlockStencilSize_;
    Foam::Info << "BVDindicator sharedMemSize = " << sharedMemSize2 << Foam::endl;
    if (sharedMemSize1 > maxSharedMemSize || sharedMemSize2 > maxSharedMemSize)
    {
        Foam::Info << "Error: Too large shared memory consumption!" << Foam::endl;
        std::exit(-1);
    }
}


void basicCUDAsolver::setDeltaT(Foam::Time& runTime)
{
    double min_deltaT = adjustTimeStep(devicePtr_, hostPtr_, facesNum_, R_, Cv_, CFL_);

    runTime.setDeltaT(min_deltaT);
}


void basicCUDAsolver::evolve(const double deltaT)
{
    GPUevolve(
        devicePtr_, hostPtr_,
        R_, Cv_, deltaT,
        cellsNum_, facesNum_, totalBoundaryFacesNum_, 
        maxStencilSize_, maxCompactStencilSize_, maxLocalBlockStencilSize_
    );
}


void basicCUDAsolver::writeField(const Foam::Time& runTime)
{
    copyFieldDataDeviceToHost(hostPtr_, devicePtr_, cellsNum_, totalBoundaryFacesNum_);

    for (int i = 0; i < cellsNum_; ++i)
    {
        pressure[i] = hostPtr_.FD[i].p;
        velocity[i].x() = hostPtr_.FD[i].Ux;
        velocity[i].y() = hostPtr_.FD[i].Uy;
        temperature[i] = hostPtr_.FD[i].T;
        shockIndicator[i] = hostPtr_.shockIndicator[i];
    }

    int tfacei = 0;
    for (int patchi = 0; patchi < pressure.boundaryField().size(); ++patchi)
    {
        Foam::fvPatchScalarField& pp = pressure.boundaryFieldRef()[patchi];
        Foam::fvPatchVectorField& pU = velocity.boundaryFieldRef()[patchi];
        Foam::fvPatchScalarField& pT = temperature.boundaryFieldRef()[patchi];
        Foam::fvPatchScalarField& pShockIndicator = shockIndicator.boundaryFieldRef()[patchi];

        if (pp.size() == 0) continue;

        for (int i = 0; i < pp.size(); ++i)
        {
            pp[i] = hostPtr_.FD[tfacei + cellsNum_].p;
            pU[i].x() = hostPtr_.FD[tfacei + cellsNum_].Ux;
            pU[i].y() = hostPtr_.FD[tfacei + cellsNum_].Uy;
            pT[i] = hostPtr_.FD[tfacei + cellsNum_].T;
            pShockIndicator[i] = 0.0;

            tfacei++;
        }
    }

    pressure.write();
    velocity.write();
    temperature.write();
    shockIndicator.write();
}