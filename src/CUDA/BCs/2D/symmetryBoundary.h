#ifndef symmetryBoundary_H
#define symmetryBoundary_H

__device__ VALUEFUNCLIST(symmetryBoundary)
{
	const double nfx = Sfx/magSf;
	const double nfy = Sfy/magSf;
	bp = pLeft;
	bUx = UxLeft - (UxLeft*nfx + UyLeft*nfy)*nfx;
	bUy = UyLeft - (UxLeft*nfx + UyLeft*nfy)*nfy;
	bT = TLeft;
}

#endif