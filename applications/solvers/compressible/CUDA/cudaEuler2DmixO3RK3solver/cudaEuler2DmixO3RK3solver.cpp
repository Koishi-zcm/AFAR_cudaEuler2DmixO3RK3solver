// #include "fvCFD.H"
#include "parRun.H"
#include "Time.H"
#include "fvMesh.H"
#include "constants.H"
#include "OSspecific.H"
#include "argList.H"
#include "timeSelector.H"

#include "basicCUDAsolver.h"

#ifndef namespaceFoam
#define namespaceFoam
    using namespace Foam;
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
int main(int argc, char *argv[])
{
    #include "setRootCaseLists.H"

    // #include "createTime.H"
    Foam::Time runTime(Foam::Time::controlDictName, args);

    // #include "createMesh.H"
    Foam::fvMesh mesh(
        Foam::IOobject(
            Foam::fvMesh::defaultRegion,
            runTime.timeName(),
            runTime,
            Foam::IOobject::MUST_READ
        )
    );

    // #include "createTimeControls.H"
    // #include "readTimeControls.H"
    // bool adjustTimeStep = runTime.controlDict().lookupOrDefault("adjustTimeStep", false);
    // scalar maxCo = runTime.controlDict().lookupOrDefault<scalar>("maxCo", 1.0);
    // scalar maxDeltaT = runTime.controlDict().lookupOrDefault<scalar>("maxDeltaT", great);
    
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    bool adjustTimeStep = readBool(runTime.controlDict().lookup("adjustTimeStep"));

    Info << "creating cudaSolver..." << endl;
    basicCUDAsolver* cudaSolver = new basicCUDAsolver(runTime, mesh);
    Info << "create cudaSolver completed." << nl << endl;

    unsigned int ITER = 0;

    Info << "\nITER    Time[s]    deltaT[s]    ExecutionTime[s]" << endl;

    while(runTime.run())
    {

        if (adjustTimeStep)
        {
            cudaSolver->setDeltaT(runTime);
        }

        runTime++;
        ITER++;

        cudaSolver->evolve(runTime.deltaT().value());

        if (runTime.write())
        {
            cudaSolver->writeField(runTime);
        }

        if (ITER % 200 == 0)
        {
            Info << "\nITER    Time[s]    deltaT[s]    ExecutionTime[s]" << endl;
        }
        if (ITER % 10 == 0)
        {
            Info << ITER << "    " << runTime.timeName() << "    " << runTime.deltaT().value() << "    " << runTime.elapsedCpuTime() << endl;
        }
    }

    Info << "\nevolve operation completed." << endl;
    Info << "ITER    Time[s]    deltaT[s]    ExecutionTime[s]" << endl;
    Info << ITER << "    " << runTime.timeName() << "    " << runTime.deltaT().value() << "    " << runTime.elapsedCpuTime() << endl;

    delete cudaSolver;

    return 0;
}
