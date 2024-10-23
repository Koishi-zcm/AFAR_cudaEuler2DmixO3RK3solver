#ifndef riemannSolver_H
#define riemannSolver_H

#include "stdlib.h"
#include "stdio.h"

__device__ void minimizeDeltaValue(float& deltaPhi, const double& phiLeft, const double& phiRight, const double& phiLeftTVD, const double& phiRightTVD)
{
    deltaPhi = fabs(phiRight - phiLeft);
    deltaPhi = fmin((double)(deltaPhi), fabs(phiRight - phiLeftTVD));
    deltaPhi = fmin((double)(deltaPhi), fabs(phiRightTVD - phiLeft));
    deltaPhi = fmin((double)(deltaPhi), fabs(phiRightTVD - phiLeftTVD));
    deltaPhi = (phiRight >= phiLeft)*deltaPhi - (phiRight < phiLeft)*deltaPhi;
}

__device__ void KFVSFlux(
    double& rhoFlux, double& rhoUxFlux, double& rhoUyFlux, double& rhoEFlux,
    const float& rhoLeft, const float& rhoRight,
    const float& pLeft, const float& pRight,
    const float& UxLeft, const float& UxRight,
    const float& UyLeft, const float& UyRight,
    const float& TLeft, const float& TRight,
    const float& deltaP, const float& deltaRho,
    const double& R, const double& Cv,
    const float magSf, const float Sfx, const float Sfy
)
{
    // evaluate local frame coordinate transform tensor
    const float nfx = Sfx/magSf;
    const float nfy = Sfy/magSf;

    // transform to local frame
    const float UxLeftLocal = UxLeft*nfx + UyLeft*nfy;
    const float UyLeftLocal = UyLeft*nfx - UxLeft*nfy;
    const float UxRightLocal = UxRight*nfx + UyRight*nfy;
    const float UyRightLocal = UyRight*nfx - UxRight*nfy;

    const float lamLeft = 0.5f*rhoLeft/pLeft;
    const float lamRight = 0.5f*rhoRight/pRight;

    const float M_u0_pos = 0.5f*erfcf(-sqrtf(lamLeft)*UxLeftLocal);
    const float M_u1_pos = UxLeftLocal*M_u0_pos + 0.5f*expf(-lamLeft*UxLeftLocal*UxLeftLocal)/(sqrtf(3.141593f*lamLeft));
    const float M_u2_pos = UxLeftLocal*M_u1_pos + 0.5f*M_u0_pos/lamLeft;
    const float M_u3_pos = UxLeftLocal*M_u2_pos + M_u1_pos/lamLeft;

    const float M_u0_neg = 0.5f*erfcf(sqrtf(lamRight)*UxRightLocal);
    const float M_u1_neg = UxRightLocal*M_u0_neg - 0.5f*expf(-lamRight*UxRightLocal*UxRightLocal)/(sqrtf(3.141593f*lamRight));
    const float M_u2_neg = UxRightLocal*M_u1_neg + 0.5f*M_u0_neg/lamRight;
    const float M_u3_neg = UxRightLocal*M_u2_neg + M_u1_neg/lamRight;

    const float& M_v1_L = UyLeftLocal;
    const float& M_v1_R = UyRightLocal;

    const float K = 2.0f*Cv/R - 2.0f;

    const float tmprhoFlux = rhoLeft*M_u1_pos + rhoRight*M_u1_neg;
    const float tmprhoUxFlux = rhoLeft*M_u2_pos + rhoRight*M_u2_neg;
    const float tmprhoUyFlux = rhoLeft*M_u1_pos*M_v1_L + rhoRight*M_u1_neg*M_v1_R;
    const float tmprhoEFlux = 0.5f*rhoLeft*(M_u3_pos + M_u1_pos*(UyLeftLocal*M_v1_L + 0.5f/lamLeft) + M_u1_pos*K*0.5f/lamLeft)
                            + 0.5f*rhoRight*(M_u3_neg + M_u1_neg*(UyRightLocal*M_v1_R + 0.5f/lamRight) + M_u1_neg*K*0.5f/lamRight);

    // transform to global frame
    rhoUxFlux += tmprhoUxFlux*Sfx - tmprhoUyFlux*Sfy - 0.5f*((float)(rhoFlux)*UxLeft + pLeft*Sfx) - 0.5f*((float)(rhoFlux)*UxRight + pRight*Sfx);
    rhoUyFlux += tmprhoUxFlux*Sfy + tmprhoUyFlux*Sfx - 0.5f*((float)(rhoFlux)*UyLeft + pLeft*Sfy) - 0.5f*((float)(rhoFlux)*UyRight + pRight*Sfy);
    rhoEFlux += tmprhoEFlux*magSf - 0.5f*(float)(rhoFlux)*(Cv*TLeft + 0.5f*(UxLeft*UxLeft + UyLeft*UyLeft) + pLeft/rhoLeft)
                                  - 0.5f*(float)(rhoFlux)*(Cv*TRight + 0.5f*(UxRight*UxRight + UyRight*UyRight) + pRight/rhoRight);
    rhoFlux += tmprhoFlux*magSf - 0.5f*rhoLeft*(UxLeft*Sfx + UyLeft*Sfy) - 0.5f*rhoRight*(UxRight*Sfx + UyRight*Sfy);
}


__device__ void RoeFlux(
    double& rhoFlux, double& rhoUxFlux, double& rhoUyFlux, double& rhoEFlux,
    const float& rhoLeft, const float& rhoRight,
    const float& pLeft, const float& pRight,
    const float& UxLeft, const float& UxRight,
    const float& UyLeft, const float& UyRight,
    const float& TLeft, const float& TRight,
    const float& deltaP, const float& deltaRho,
    const double& R, const double& Cv,
    const float magSf, const float Sfx, const float Sfy
)
{
    const float gamma = (Cv + R)/Cv;

    const float nfx = Sfx/magSf;
    const float nfy = Sfy/magSf;

    const float contrVLeft  = UxLeft*nfx + UyLeft*nfy;
    const float contrVRight = UxRight*nfx + UyRight*nfy;

    const float hLeft = Cv*TLeft + 0.5f*(UxLeft*UxLeft + UyLeft*UyLeft) + pLeft/rhoLeft;
    const float hRight = Cv*TRight + 0.5f*(UxRight*UxRight + UyRight*UyRight) + pRight/rhoRight;

    const float rhoTilde = sqrt(max(rhoLeft*rhoRight, 1e-20f));

    const float wLeft = sqrt(max(rhoLeft, 1e-20f))/(sqrt(max(rhoLeft, 1e-20f)) + sqrt(max(rhoRight, 1e-20f)));
    const float wRight = 1.0f - wLeft;

    const float UxTilde = UxLeft*wLeft + UxRight*wRight;
    const float UyTilde = UyLeft*wLeft + UyRight*wRight;
    const float hTilde = hLeft*wLeft + hRight*wRight;

    const float qTildeSquare = UxTilde*UxTilde + UyTilde*UyTilde;
    const float cTilde = sqrt(max((gamma - 1.0f)*(hTilde - 0.5f*qTildeSquare), 1e-20f));
    const float contrVTilde = UxTilde*nfx + UyTilde*nfy;

    const float deltaUx = UxRight - UxLeft;
    const float deltaUy = UyRight - UyLeft;
    const float deltaContrV = deltaUx*nfx + deltaUy*nfy;

    const float r1 = (deltaP - rhoTilde*cTilde*deltaContrV)/(2.0f*cTilde*cTilde);
    const float r2 = deltaRho - deltaP/(cTilde*cTilde);
    const float r3 = (deltaP + rhoTilde*cTilde*deltaContrV)/(2.0f*cTilde*cTilde);

    const float UL = UxLeft*nfx + UyLeft*nfy;
    const float UR = UxRight*nfx + UyRight*nfy;
    const float cLeft = sqrt( max((gamma - 1.0f)*(hLeft - 0.5f*(UxLeft*UxLeft + UyLeft*UyRight)), 1e-20f) );
    const float cRight = sqrt( max((gamma - 1.0f)*(hRight - 0.5f*(UxRight*UxRight + UyRight*UyRight)), 1e-20f) );

    float eps = 2.0f*max(0.0f, (UR - cRight) - (UL - cLeft));
    float lambda1 = abs(contrVTilde - cTilde);
    lambda1 = (lambda1 >= eps)*lambda1 + (lambda1 < eps)*( (lambda1*lambda1 + eps*eps)/(2.0f*eps + 1e-30f) );

    eps = 2.0f*max(0.0f, UR - UL);
    float lambda2 = abs(contrVTilde);
    lambda2 = (lambda2 >= eps)*lambda2 + (lambda2 < eps)*( (lambda2*lambda2 + eps*eps)/(2.0f*eps + 1e-30f) );

    eps = 2.0f*max(0.0f, (UR + cRight) - (UL + cLeft));
    float lambda3 = abs(contrVTilde + cTilde);
    lambda3 = (lambda3 >= eps)*lambda3 + (lambda3 < eps)*( (lambda3*lambda3 + eps*eps)/(2.0f*eps + 1e-30f) );

    rhoFlux -= 0.5f*(lambda1*r1 + lambda2*r2 + lambda3*r3)*magSf;
    rhoUxFlux -= 0.5f*(lambda1*r1*(UxTilde - cTilde*nfx) + lambda2*(r2*UxTilde + rhoTilde*(deltaUx - deltaContrV*nfx)) + lambda3*r3*(UxTilde + cTilde*nfx))*magSf;
    rhoUyFlux -= 0.5f*(lambda1*r1*(UyTilde - cTilde*nfy) + lambda2*(r2*UyTilde + rhoTilde*(deltaUy - deltaContrV*nfy)) + lambda3*r3*(UyTilde + cTilde*nfy))*magSf;
    rhoEFlux -= 0.5f*(lambda1*r1*(hTilde - cTilde*contrVTilde) + lambda2*(r2*0.5f*qTildeSquare + rhoTilde*(UxTilde*deltaUx + UyTilde*deltaUy - contrVTilde*deltaContrV)) + lambda3*r3*(hTilde + cTilde*contrVTilde))*magSf;
}


__device__ void (*calcFlux[])(
    double& rhoFlux, double& rhoUxFlux, double& rhoUyFlux, double& rhoEFlux,
    const float& rhoLeft, const float& rhoRight,
    const float& pLeft, const float& pRight,
    const float& UxLeft, const float& UxRight,
    const float& UyLeft, const float& UyRight,
    const float& TLeft, const float& TRight,
    const float& deltaP, const float& deltaRho,
    const double& R, const double& Cv,
    const float magSf, const float Sfx, const float Sfy
) = {
    RoeFlux,
    KFVSFlux
};


__global__ void evaluateFlux(
    basicFluxData* __restrict__ Flux,
    const basicFieldData* __restrict__ FD,
    const gradientFieldData* __restrict__ gradFD,
    const limiterFieldData* __restrict__ limiter,
    const basicFieldData* __restrict__ faceFD,
    const int8_t* __restrict__ shockIndicator,
    const meshCellData* __restrict__ CELL,
    const meshFaceData* __restrict__ FACE,
    const int* __restrict__ neighbour,
    const int facesNum,
    const int cellsNum,
    const int totalBoundaryFacesNum,
    const double R, const double Cv
)
{
    const int faceI = threadIdx.x + blockIdx.x*blockDim.x;

    if(faceI < facesNum)
    {
        const int own = neighbour[2*faceI];
        const int nei = neighbour[2*faceI+1];

        const double deltaRxLeft = FACE[faceI].x - CELL[own].x;
        const double deltaRyLeft = FACE[faceI].y - CELL[own].y;

        const double deltaRxRight = FACE[faceI].x - CELL[nei].x;
        const double deltaRyRight = FACE[faceI].y - CELL[nei].y;

        const bool isTVD_left = shockIndicator[own] != 0;
        const bool isTVD_right = shockIndicator[nei] != 0;
        const bool isShock = shockIndicator[own] < 0 || shockIndicator[nei] < 0;

        const double magSf = FACE[faceI].magSf;
        const double Sfx = FACE[faceI].Sfx;
        const double Sfy = FACE[faceI].Sfy;

        double rhoFlux, rhoUxFlux, rhoUyFlux, rhoEFlux;
        float pLeftf, UxLeftf, UyLeftf, TLeftf, rhoLeftf;
        float pRightf, UxRightf, UyRightf, TRightf, rhoRightf;
        float deltaP, deltaRho;

        {
            const double pLeftTVD = FD[own].p + limiter[own].pLimiter*(gradFD[own].gradPx*deltaRxLeft + gradFD[own].gradPy*deltaRyLeft);
            const double UxLeftTVD = FD[own].Ux + limiter[own].UxLimiter*(gradFD[own].gradUxx*deltaRxLeft + gradFD[own].gradUyx*deltaRyLeft);
            const double UyLeftTVD = FD[own].Uy + limiter[own].UyLimiter*(gradFD[own].gradUxy*deltaRxLeft + gradFD[own].gradUyy*deltaRyLeft);
            const double TLeftTVD = FD[own].T + limiter[own].TLimiter*(gradFD[own].gradTx*deltaRxLeft + gradFD[own].gradTy*deltaRyLeft);
            const double rhoLeftTVD = pLeftTVD/(R*TLeftTVD);

            const double pRightTVD = FD[nei].p + limiter[nei].pLimiter*(gradFD[nei].gradPx*deltaRxRight + gradFD[nei].gradPy*deltaRyRight);
            const double UxRightTVD = FD[nei].Ux + limiter[nei].UxLimiter*(gradFD[nei].gradUxx*deltaRxRight + gradFD[nei].gradUyx*deltaRyRight);
            const double UyRightTVD = FD[nei].Uy + limiter[nei].UyLimiter*(gradFD[nei].gradUxy*deltaRxRight + gradFD[nei].gradUyy*deltaRyRight);
            const double TRightTVD = FD[nei].T + limiter[nei].TLimiter*(gradFD[nei].gradTx*deltaRxRight + gradFD[nei].gradTy*deltaRyRight);
            const double rhoRightTVD = pRightTVD/(R*TRightTVD);

            const double pLeft = (1 - isTVD_left)*faceFD[2*faceI].p + isTVD_left*pLeftTVD;
            const double UxLeft = (1 - isTVD_left)*faceFD[2*faceI].Ux + isTVD_left*UxLeftTVD;
            const double UyLeft = (1 - isTVD_left)*faceFD[2*faceI].Uy + isTVD_left*UyLeftTVD;
            const double TLeft = (1 - isTVD_left)*faceFD[2*faceI].T + isTVD_left*TLeftTVD;
            const double rhoLeft = pLeft/(R*TLeft);

            const double pRight = (1 - isTVD_right)*faceFD[2*faceI+1].p + isTVD_right*pRightTVD;
            const double UxRight = (1 - isTVD_right)*faceFD[2*faceI+1].Ux + isTVD_right*UxRightTVD;
            const double UyRight = (1 - isTVD_right)*faceFD[2*faceI+1].Uy + isTVD_right*UyRightTVD;
            const double TRight = (1 - isTVD_right)*faceFD[2*faceI+1].T + isTVD_right*TRightTVD;
            const double rhoRight = pRight/(R*TRight);

            minimizeDeltaValue(deltaP, pLeft, pRight, pLeftTVD, pRightTVD);
            minimizeDeltaValue(deltaRho, rhoLeft, rhoRight, rhoLeftTVD, rhoRightTVD);

            rhoFlux = 0.5*rhoLeft*(UxLeft*Sfx + UyLeft*Sfy) + 0.5*rhoRight*(UxRight*Sfx + UyRight*Sfy);
            rhoUxFlux = 0.5*(rhoFlux*UxLeft + pLeft*Sfx) + 0.5*(rhoFlux*UxRight + pRight*Sfx);
            rhoUyFlux = 0.5*(rhoFlux*UyLeft + pLeft*Sfy) + 0.5*(rhoFlux*UyRight + pRight*Sfy);
            rhoEFlux = 0.5*rhoFlux*(Cv*TLeft + 0.5*(UxLeft*UxLeft + UyLeft*UyLeft) + pLeft/rhoLeft)
                     + 0.5*rhoFlux*(Cv*TRight + 0.5*(UxRight*UxRight + UyRight*UyRight) + pRight/rhoRight);

            pLeftf = pLeft;  UxLeftf = UxLeft;  UyLeftf = UyLeft;  TLeftf = TLeft;  rhoLeftf = rhoLeft;
            pRightf = pRight;  UxRightf = UxRight;  UyRightf = UyRight;  TRightf = TRight;  rhoRightf = rhoRight;
        }

        calcFlux[isShock](
            rhoFlux, rhoUxFlux, rhoUyFlux, rhoEFlux,
            rhoLeftf, rhoRightf,
            pLeftf, pRightf,
            UxLeftf, UxRightf,
            UyLeftf, UyRightf,
            TLeftf, TRightf,
            deltaP, deltaRho,
            R, Cv,
            magSf, Sfx, Sfy
        );

        Flux[faceI].rhoFlux = rhoFlux;
        Flux[faceI].rhoUxFlux = rhoUxFlux;
        Flux[faceI].rhoUyFlux = rhoUyFlux;
        Flux[faceI].rhoEFlux = rhoEFlux;
    }


    if (faceI < totalBoundaryFacesNum)
    {
        const double Sfx = FACE[facesNum + faceI].Sfx;
        const double Sfy = FACE[facesNum + faceI].Sfy;

        const double bp = FD[cellsNum + faceI].p;
        const double bUx = FD[cellsNum + faceI].Ux;
        const double bUy = FD[cellsNum + faceI].Uy;
        const double bT = FD[cellsNum + faceI].T;
        const double rho = bp/(R*bT);

        const double rhoFlux = rho*(bUx*Sfx + bUy*Sfy);
        Flux[facesNum + faceI].rhoFlux = rhoFlux;
        Flux[facesNum + faceI].rhoUxFlux = rhoFlux*bUx + bp*Sfx;
        Flux[facesNum + faceI].rhoUyFlux = rhoFlux*bUy + bp*Sfy;
        Flux[facesNum + faceI].rhoEFlux = rhoFlux*(Cv*bT + 0.5*(bUx*bUx + bUy*bUy) + bp/rho);
    }
}

#endif