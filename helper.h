#include <thrust/device_vector.h>

void initializeToMaxInt(uint32_t * du, uint32_t Ndofs) 
{
    for (int i = 0; i < Ndofs; i++) {
        du[i] = UINT32_MAX;
    }
}


void initializeToZerosInt(uint32_t * du, uint32_t Ndofs) 
{
    for (int i = 0; i < Ndofs; i++) {
        du[i] = 0;
    }
}

void initializeToZeros(float * du, uint32_t Ndofs) 
{
    for (int i = 0; i < Ndofs; i++) {
        du[i] = 0.0;
    }
}

void initializeToOnes(float * du, uint32_t Ndofs) 
{
    for (int i = 0; i < Ndofs; i++) {
        du[i] = 0.0;
    }
}

////////////////////////////
/*
allocateDataOnDevice(struct matrix) 
{

}

copyDataToDevice(struct matrix)
{

}

initializeLinearSystemHost(struct linearSystemInfo *lS)
{
	uint32_t Ndofs = lS.Ndofs;
	lS.du0 = new float[Ndofs];
	lS.du1 = new float[Ndofs];
	lS.rhs = new float[Ndofs];
	initializeToZeros(lS.du0, Ndofs);
	initializeToZeros(lS.du1, Ndofs);
	initializeToZeros(lS.rhs, Ndofs);
}

allocateLinearSystemDevice(struct linearSystemInfo *lS)
{

}
*/
