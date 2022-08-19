__global__
void computeResidualVectorDevice(float * residualVector, float * solution, linearSystemDevice matrix)
{
	// Define the ID of DOF the thread will handle
	uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
	
	// Ensure update index does not go beyond actual matrix/mesh size
	if (i < matrix.Ndofs) {	

		// Fill in vector containing b - Ax
		residualVector[i] = matrix.rhs_d[i];
		for (int j = matrix.indexPtr_d[i]; j < matrix.indexPtr_d[i+1]; j++) {
			uint32_t n = matrix.nodeNeighbors_d[j];
			residualVector[i] -= matrix.offdiags_d[j] * solution[n];
			// printf("For dof %d, global neighbor %d has weight %f\n", i, n, matrix.offdiags_d[j]);
		}
		__syncthreads();

		residualVector[i] -= (1.0 / matrix.diagInv_d[i]) * solution[i];
		// printf("The D^{-1} term is %f\n", matrix.diagInv_d[i]);

		// Square each entry in residual vector (preparation for L2 norm calculation)
		residualVector[i] = residualVector[i] * residualVector[i];
	}

}

float computeL2Residual(float * solution, linearSystemDevice matrix) 
{
	// Initial L2 residual norm
	float L2residual = 0.0;

	// Create residualVector containers on host and device
	float * residualVector = new float[matrix.Ndofs];
	float * residualVector_d;
	cudaMalloc(&residualVector_d, sizeof(float) * matrix.Ndofs);

	// Compute residualVector (squared elements) on the GPU
	uint32_t tpb = 32;
	uint32_t numBlocks = ceil((float)matrix.Ndofs / tpb);
	computeResidualVectorDevice<<<numBlocks, tpb>>>(residualVector_d, solution, matrix);
	cudaDeviceSynchronize();

	// Copy residual vector from device to host
	cudaMemcpy(residualVector, residualVector_d, sizeof(float) * matrix.Ndofs, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// Compute L2 residual norm
	for (int i = 0; i < matrix.Ndofs; i++) {
		L2residual += residualVector[i];
		// printf("residualVector[%d] = %f\n", i, sqrt(residualVector[i])); 
	}
	L2residual = sqrt(L2residual);

	return L2residual;
}
