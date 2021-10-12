__global__
void globalMemoryIteration(float * du1, float * du0, uint32_t N)
{
    // Initialize variables
    uint32_t i, j, gID;
    uint32_t left, right, top, bottom;

    // Identify (i,j) coordinates corresponding to thread 
    i = threadIdx.x + blockDim.x * blockIdx.x;
    j = threadIdx.y + blockDim.y * blockIdx.y;

    // Perform update within domain bounds
    if ((i < N) && (j < N)) {
    	// Compute the global DOF ID number
    	gID = i + j * N;
        // Define neighbor values (considering boundary conditions)
	    left = (i > 0) ? du0[gID-1] : 0.0;
        right = (i < N-1) ? du0[gID+1] : 0.0;
        top = (j < N-1) ? du0[gID+N] : 0.0;
        bottom = (j > 0) ? du0[gID-N] : 0.0;
    	// Perform the Jacobi update for uniform 2D Poisson
    	du1[gID] = (left + right + top + bottom) / 4;
    }
}

void globalMemorySolve2D(float * du1, float * du0, uint32_t N, uint32_t numIters)
{
    // Define the number of threads per block in each direction
    uint32_t tpbx = 32;
    uint32_t tpby = 32;
    dim3 threadsPerBlock(tpbx, tpby);

    // Define the number of blocks in each direction
    uint32_t nBlocksx = ceil((float)N / tpbx);
    uint32_t nBlocksy = ceil((float)N / tpby);
    dim3 blocks(nBlocksx, nBlocksy);

    // Create a temporary variable for the solution swap
    float * tmp;

    // Perform all Jacobi iterations
    for (int iter = 0; iter < numIters; iter++) {
        // Perform stencil update on global memory
        globalMemoryIteration<<<blocks, threadsPerBlock>>>(du1, du0, N);
        // Update the solution 
        tmp = du0; du0 = du1; du1 = tmp;
    }

}

