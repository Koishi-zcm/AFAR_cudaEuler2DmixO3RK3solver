#ifndef constantTemperatureWallBoundary_H
#define constantTemperatureWallBoundary_H

__device__ VALUEFUNCLIST(constantTemperatureWallBoundary)
{
	bp = pLeft;
	bUx = 0.0;
	bUy = 0.0;
	bT = 1.0;
}

#endif