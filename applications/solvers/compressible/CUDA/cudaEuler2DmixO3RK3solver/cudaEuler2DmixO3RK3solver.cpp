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

    Info << "\nITER\tTime[s]\tdeltaT[s]\tExecutionTime[s]" << endl;

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
            Info << "\nITER\tTime[s]\tdeltaT[s]\tExecutionTime[s]" << endl;
        }
        if (ITER % 10 == 0)
        {
            Info << ITER << "\t" << runTime.timeName() << "\t" << runTime.deltaT().value() << "\t" << runTime.elapsedCpuTime() << endl;
        }
    }

    Info << "\nevolve operation completed." << endl;
    Info << "ITER\tTime[s]\tdeltaT[s]\tExecutionTime[s]" << endl;
    Info << ITER << "\t" << runTime.timeName() << "\t" << runTime.deltaT().value() << "\t" << runTime.elapsedCpuTime() << endl;

    delete cudaSolver;

    return 0;
}
