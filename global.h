/* 2D Structured Functions */

__global__
void globalMemoryIteration(float * du1, float * du0, linearSystemDevice matrix, uint32_t N)
{
    // Initialize variables
    uint32_t i, j, g;
    // uint32_t left, right, top, bottom;

    // Identify (i,j) coordinates corresponding to thread 
    i = threadIdx.x + blockDim.x * blockIdx.x;
    j = threadIdx.y + blockDim.y * blockIdx.y;

    // Perform update within domain bounds
    if ((i < N) && (j < N)) {
    	// Compute the global DOF ID number
    	g = i + j * N;
        // Averaging kernel
		/*
		// Define neighbor values (considering boundary conditions)
	    left = (i > 0) ? du0[g-1] : 0.0;
        right = (i < N-1) ? du0[g+1] : 0.0;
        top = (j < N-1) ? du0[g+N] : 0.0;
        bottom = (j > 0) ? du0[g-N] : 0.0;
    	// Perform the Jacobi update for uniform 2D Poisson
    	du1[g] = (left + right + top + bottom) / 4;
		*/
		// Jacobi
		du1[g] = matrix.rhs_d[g];
		for (int k = matrix.indexPtr_d[g]; k < matrix.indexPtr_d[g+1]; k++) {
			uint32_t n = matrix.nodeNeighbors_d[k];
			du1[g] -= matrix.offdiags_d[k] * du0[n];
		}	
		du1[g] *= matrix.diagInv_d[g];
    }
}

void globalMemorySolve2D(float * du1, float * du0, linearSystemDevice matrix, uint32_t N, uint32_t numIters)
{
    // Define the number of threads per block in each direction
    uint32_t tpbx = 16;
    uint32_t tpby = 16;
    dim3 threadsPerBlock(tpbx, tpby);

    // Define the number of blocks in each direction
    uint32_t nBlocksx = ceil((float)N / tpbx);
    uint32_t nBlocksy = ceil((float)N / tpby);
    dim3 blocks(nBlocksx, nBlocksy);

    // Create a temporary variable for the solution swap

    // Perform all Jacobi iterations
    for (int iter = 0; iter < numIters; iter++) {
        // Perform stencil update on global memory
        globalMemoryIteration<<<blocks, threadsPerBlock>>>(du1, du0, matrix, N);
		// Synchronize
		cudaDeviceSynchronize();
        // Update the solution 
    	float * tmp; tmp = du0; du0 = du1; du1 = tmp;
    }
}

/* 2D Unstructured Functions */

__global__
void globalMemoryIterationUnstructured(float * du1, float * du0, linearSystemDevice matrix, uint32_t iter)
{
    // Initialize variables
    uint32_t i;

    // Identify (i,j) coordinates corresponding to thread 
    i = threadIdx.x + blockDim.x * blockIdx.x;

    // Perform update within domain bounds
    if (i < matrix.Ndofs) {
		// Jacobi
		du1[i] = matrix.rhs_d[i];
		for (int k = matrix.indexPtr_d[i]; k < matrix.indexPtr_d[i+1]; k++) {
			uint32_t n = matrix.nodeNeighbors_d[k];
			du1[i] -= matrix.offdiags_d[k] * du0[n];
		}	
		du1[i] *= matrix.diagInv_d[i];
    }
}

void globalMemorySolveUnstructured(float * du1, float * du0, linearSystemDevice matrix, uint32_t numIters)
{
    // Define the number of threads per block in each direction
    uint32_t threadsPerBlock = 128;
    uint32_t numBlocks = ceil((float)matrix.Ndofs / threadsPerBlock);

    // Perform all Jacobi iterations
    for (int iter = 0; iter < numIters; iter++) {
        // Perform stencil update on global memory
        globalMemoryIterationUnstructured<<<numBlocks, threadsPerBlock>>>(du1, du0, matrix, iter);
		// Synchronize
		cudaDeviceSynchronize();
        // Update the solution 
    	float * tmp; tmp = du0; du0 = du1; du1 = tmp;
    }
}

