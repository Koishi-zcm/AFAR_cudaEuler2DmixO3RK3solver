#ifndef adiabaticWallBoundary_H
#define adiabaticWallBoundary_H

__device__ VALUEFUNCLIST(adiabaticWallBoundary)
{
	bp = pLeft;
	bUx = 0.0;
	bUy = 0.0;
	bT = TLeft;
}

#endif