// CPU Host Initialize

// Int

void initializeToZerosInt(uint32_t * du, uint32_t Ndofs) 
{
    for (int i = 0; i < Ndofs; i++) {
        du[i] = 0;
    }
}

void initializeToMaxInt(uint32_t * du, uint32_t Ndofs) 
{
    for (int i = 0; i < Ndofs; i++) {
        du[i] = UINT32_MAX;
    }
}

// Float

void initializeToZeros(float * du, uint32_t Ndofs) 
{
    for (int i = 0; i < Ndofs; i++) {
        du[i] = 0.0;
    }
}

void initializeToOnes(float * du, uint32_t Ndofs) 
{
    for (int i = 0; i < Ndofs; i++) {
        du[i] = 1.0;
    }
}

// GPU Device Initialize

// Int

__global__
void initializeToZerosDeviceInt(uint32_t * du, uint32_t Ndofs) 
{
	uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < Ndofs) {
		du[i] = 0;
    }
}

// Float

__global__
void initializeToZerosDevice(float * du, uint32_t Ndofs) 
{
	uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < Ndofs) {
		du[i] = 0.0;
    }
}



