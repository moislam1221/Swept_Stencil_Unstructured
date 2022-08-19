__global__
void orderSolutionVectorBySubdomain(float * solutionOrdered, float * solution, uint32_t * territoryDOFs, uint32_t Ndofs)
{
	// Initialize index related parameters
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t index;

	// Create an ordered solution which follows the same ordering of DOFs based on subdomain
	if (i < Ndofs) {
		index = territoryDOFs[i];
		solutionOrdered[i] = solution[index];
	}	
}

__global__
void undoOrderSolutionVectorBySubdomain(float * solution, float * solutionBySubdomain, meshPartitionForStage partition, uint32_t Ndofs)
{
	// Initialize index related parameters
	uint32_t i = threadIdx.x;
	uint32_t idx, index;
	bool update;
	uint32_t idx_lower_inner = partition.territoryIndexPtr_d[blockIdx.x];
	uint32_t idx_upper_inner = partition.territoryIndexPtr_d[blockIdx.x + 1];
	uint32_t idx_lower = partition.territoryIndexPtrExpanded_d[blockIdx.x];
	uint32_t idx_upper = partition.territoryIndexPtrExpanded_d[blockIdx.x + 1];

	// Create an ordered solution which follows the same ordering of DOFs based on subdomain
	if (i < idx_upper - idx_lower) {
		idx = i + idx_lower;
		index = partition.territoryDOFsExpanded_d[idx];
		update = false;
		for (int j = idx_lower_inner; j < idx_upper_inner; j++) {
			if (partition.territoryDOFsExpanded_d[idx] == partition.territoryDOFs_d[j]) {
				update = true;
				j = idx_upper_inner; // end the for loop
			}
		}
		if (update == true) {
			solution[index] = solutionBySubdomain[idx];
		}
	}
}

__global__
void stageAdvance(float * solution, uint32_t * iterationLevels, meshPartitionForStage partition, matrixInfo matrix, uint32_t numExpansionSteps)
{
	extern __shared__ float sharedMemorySolution[];

	// 1 - Move solution from global memory to shared memory
	uint32_t idx_lower, idx_upper;
	idx_lower = partition.territoryIndexPtrExpanded_d[blockIdx.x];
	idx_upper = partition.territoryIndexPtrExpanded_d[blockIdx.x + 1];
	uint32_t i = threadIdx.x;
	if (i < idx_upper - idx_lower) {
		sharedMemorySolution[i] = solution[i + idx_lower];
	}
	__syncthreads();
	
	// 2 - Perform updates within shared memory
	uint32_t idx, dof, neighbor, myGroundLevel;
	bool updateDOF, isMember;
	if (i < idx_upper - idx_lower) {
	
		for (int iter = 0; iter < numExpansionSteps; iter++) {
			idx = threadIdx.x + idx_lower;
			dof = partition.territoryDOFsExpanded_d[idx];
			myGroundLevel = iterationLevels[dof];
			updateDOF = true;
			for (int j = matrix.indexPtr_d[dof]; j < matrix.indexPtr_d[dof+1]; j++) {
				neighbor = matrix.nodeNeighbors_d[j];
				// if neighbor is not a member of all dofs in the set, duplicate DOF IS false
				isMember = false;
				for (int k = idx_lower; k < idx_upper; k++) {
					if (neighbor == partition.territoryDOFsExpanded_d[k]) {
						isMember = true;
					}
				}
				if (isMember == false) {
					updateDOF = false;
				}
			}
			__syncthreads();
			if (updateDOF == true) {
				if ((int)myGroundLevel < (iter + 1 - (int)partition.distanceFromSeed_d[dof]) && partition.distanceFromSeed_d[dof] != UINT32_MAX) {
					sharedMemorySolution[i] += 1;
					iterationLevels[dof] += 1;
					// printf("In iter %d, updating dof %d\n", iter, dof);
				}
			} 
		__syncthreads();
		}
	}
	__syncthreads();

	// 3 - Return solution from shared memory to global memory
	// for (int i = threadIdx.x; i < idx_upper - idx_lower; i += blockDim.x) {
	if (i < idx_upper - idx_lower) {
		solution[i + idx_lower] = sharedMemorySolution[i];
	}
	__syncthreads();
}

void advanceFunction(float * solution_d, uint32_t * iterationLevel_d, meshPartitionForStage &partition, matrixInfo matrix, uint32_t numExpansionSteps)
{
	// Call function to reorder the solution vector in preparation for transfer to shared memory
	uint32_t numElemsExpanded = partition.territoryIndexPtrExpanded[partition.numSubdomains];
    float * solutionBySubdomain_d;
    cudaMalloc(&solutionBySubdomain_d, sizeof(float) * numElemsExpanded);
	uint32_t threadsPerBlock = 256;
	uint32_t numBlocks = ceil((float)numElemsExpanded / threadsPerBlock);

	// Perform the update
	// 1 - Create the solution vector partitioned between different blocks sol = [ block 1 | block 2 | block 3 | etc. ] 
	cudaDeviceSynchronize();
	orderSolutionVectorBySubdomain<<<numBlocks, threadsPerBlock>>>(solutionBySubdomain_d, solution_d, partition.territoryDOFsExpanded_d, numElemsExpanded);
	cudaDeviceSynchronize();

	// 2 - Perform Jacobi updates in shared memory 
	numBlocks = partition.numSubdomains;
	stageAdvance<<<numBlocks, threadsPerBlock, 4 * sizeof(float) * (matrix.Ndofs/4)>>>(solutionBySubdomain_d, iterationLevel_d, partition, matrix, numExpansionSteps);
	cudaDeviceSynchronize();

	// 3 - Transfer solution vector per subdomain into original solution vector 
	undoOrderSolutionVectorBySubdomain<<<numBlocks, threadsPerBlock>>>(solution_d, solutionBySubdomain_d, partition, matrix.Ndofs);
	cudaDeviceSynchronize();
}
