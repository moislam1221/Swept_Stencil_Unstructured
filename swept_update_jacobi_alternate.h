























__global__
void stageAdvanceJacobiPerformanceSolutionOnly(float * evenSolutionBuffer, float * oddSolutionBuffer, uint32_t * iterationLevels, meshPartitionForStageDevice partition, uint32_t numJacobiStepsMin, uint32_t numJacobiStepsMax, uint32_t maxIterShift = 0, bool finalStage = false)
{
	extern __shared__ float sharedMemorySolution[];
	
	// 1 - Move solution from global memory to shared memory
	uint32_t idx_lower, idx_upper, numDOFs;
	uint32_t i, g;
	// Define bounds of the dofs for subdomain (e.g. (0, 9) or (9,18) of territoryDOFs)
	idx_lower = partition.territoryIndexPtr_d[blockIdx.x];
	idx_upper = partition.territoryIndexPtr_d[blockIdx.x + 1];
	numDOFs = idx_upper - idx_lower;
	// Define two portions of the shared memory solution for Jacobi
	float * du0 = sharedMemorySolution;
	float * du1 = sharedMemorySolution + numDOFs;
	for (int i = threadIdx.x; i < numDOFs; i+= blockDim.x) {
		// Define local index i within block, and global index g
		// i = threadIdx.x;
		g = partition.territoryDOFs_d[idx_lower + i];
		if (numJacobiStepsMin % 2 == 0) {
			du0[i] = evenSolutionBuffer[g];
			du1[i] = oddSolutionBuffer[g];
		}
		else {
			du0[i] = oddSolutionBuffer[g];
			du1[i] = evenSolutionBuffer[g];
		}
	}
	// Pointer to matrix and to local rhs and diagInv
	uint32_t shiftIndexPtr = partition.indexPtrIndexPtrSubdomain_d[blockIdx.x];
	uint32_t shiftData = partition.indexPtrDataShiftSubdomain_d[blockIdx.x];
	uint32_t * localIndexPtr = partition.indexPtrSubdomain_d + shiftIndexPtr;
	uint32_t * localNodeNeighbors = partition.nodeNeighborsSubdomain_d + shiftData;
	float * localOffDiags = partition.offDiagsSubdomain_d + shiftData;
	float * localRhs = partition.rhsLocal_d + idx_lower;
	float * localDiagInv = partition.diagInvLocal_d + idx_lower;
	// Define nnz
	uint32_t nnz = partition.indexPtrDataShiftSubdomain_d[blockIdx.x+1] - partition.indexPtrDataShiftSubdomain_d[blockIdx.x];
	uint32_t numDOFsToUpdate = partition.numDOFsInteriorPerSubdomain_d[blockIdx.x]; // partition.territoryIndexPtrInteriorExt_d[blockIdx.x+1] - partition.territoryIndexPtrInteriorExt_d[blockIdx.x];
	// Fill in shared memory pointers
	/*
	for (int i = threadIdx.x; i < nnz; i+= blockDim.x) {
		if (i < idx_upper - idx_lower) {
			sharedDiagInv[i] = localDiagInv[i];
			sharedRhs[i] = localRhs[i];
		}
		if (i < numDOFsToUpdate+1) {
			sharedIndexPtr[i] = localIndexPtr[i];
		}
		sharedOffDiags[i] = localOffDiags[i];
		sharedNodeNeighbors[i] = localNodeNeighbors[i];
	} */
	__syncthreads();

	// 2 - Perform updates within shared memory
	// numDOFsToUpdate = partition.territoryIndexPtrInteriorExt_d[blockIdx.x+1] - partition.territoryIndexPtrInteriorExt_d[blockIdx.x];
	uint32_t lowerIdx = partition.territoryIndexPtrInterior_d[blockIdx.x];
	i = threadIdx.x;
	if (i < numDOFsToUpdate) {
		g = partition.territoryDOFs_d[idx_lower + i];
		if (finalStage == 0) {
			for (int iter = numJacobiStepsMin; iter < numJacobiStepsMax; iter++) {
				// printf("iter = %d\n", iter);
				if (iter >= iterationLevelShared[i] && iter < partition.iterationLevelDOFs_d[lowerIdx + i] + maxIterShift) {
					iterationLevelShared[i] += 1;
					// Jacobi with shared data structures
					/* du1[i] = sharedRhs[i]; // shared
					for (int k = sharedIndexPtr[i]; k < sharedIndexPtr[i+1]; k++) { // shared
						uint32_t n = sharedNodeNeighbors[k]; // shared
						du1[i] -= sharedOffDiags[k] * du0[n]; // shared
					} 
					du1[i] *= sharedDiagInv[i]; // shared */
					du1[i] = localRhs[i]; // shared
					for (int k = localIndexPtr[i]; k < localIndexPtr[i+1]; k++) { // shared
						uint32_t n = localNodeNeighbors[k]; // shared
						du1[i] -= localOffDiags[k] * du0[n]; // shared
					} 
					du1[i] *= localDiagInv[i]; // shared */
				} 	
				float * tmp; tmp = du0; du0 = du1; du1 = tmp;
				__syncthreads();
			}
		}
		else {
			for (int iter = numJacobiStepsMin; iter < numJacobiStepsMax; iter++) {
				if (iter >= iterationLevelShared[i]) {
					iterationLevelShared[i] += 1;
					/* // Jacobi with shared data structures
					du1[i] = sharedRhs[i]; // shared
					for (int k = sharedIndexPtr[i]; k < sharedIndexPtr[i+1]; k++) { // shared
						uint32_t n = sharedNodeNeighbors[k]; // shared
						du1[i] -= sharedOffDiags[k] * du0[n]; // shared
					} 
					du1[i] *= sharedDiagInv[i]; // shared */
					du1[i] = localRhs[i]; // shared
					for (int k = localIndexPtr[i]; k < localIndexPtr[i+1]; k++) { // shared
						uint32_t n = localNodeNeighbors[k]; // shared
						du1[i] -= localOffDiags[k] * du0[n]; // shared
					} 
					du1[i] *= localDiagInv[i]; // shared */
				} 	
				float * tmp; tmp = du0; du0 = du1; du1 = tmp;
				__syncthreads();
			}
		}
	}
	__syncthreads();

	// 3 - Return solution from shared memory to global memory
	i = threadIdx.x;
	// uint32_t numDOFsInterior =  partition.territoryIndexPtrInterior_d[blockIdx.x + 1] - partition.territoryIndexPtrInterior_d[blockIdx.x];
	uint32_t numDOFsToReturn = partition.numDOFsToReturnPerSubdomain_d[blockIdx.x];
	if (i < numDOFsToReturn) {
		// printf("interior DOFs[%d] = %d\n", blockIdx.x, partition.interiorDOFsPerSubdomain_d[blockIdx.x]);
		g = partition.territoryDOFs_d[idx_lower + i];
		iterationLevels[g] = iterationLevelShared[i];
		// printf("blockIdx.x = %d: global dof = %d, num dofs to return = %d\n", blockIdx.x, g, numDOFsToReturn);
		if (numJacobiStepsMax % 2 == 0) {
			evenSolutionBuffer[g] = du0[i];
			oddSolutionBuffer[g] = du1[i];
		}
		else if (numJacobiStepsMax % 2 == 1) {
			evenSolutionBuffer[g] = du1[i];
			oddSolutionBuffer[g] = du0[i];
		}
	}
	__syncthreads();
}

void determineSharedMemoryAllocationSolutionOnly(meshPartitionForStage &partition)
{
	// Determine the appropriate shared memory allocation (sum of twice the subdomain DOFs plus number of nnz entries for offdiags)
	uint32_t sharedMemorySize = 0;
	uint32_t sharedMemorySizeTemp;
	uint32_t numDOFs;
	for (int i = 0; i < partition.numSubdomains; i++) {
		numDOFs = partition.territoryIndexPtr[i+1] - partition.territoryIndexPtr[i];
		sharedMemorySizeTemp = 3 * numDOFs * sizeof(float);
		if (sharedMemorySize < sharedMemorySizeTemp) {
			sharedMemorySize = sharedMemorySizeTemp;
		}
	}
	partition.sharedMemorySize = sharedMemorySize;
	printf("The shared memory size is %d\n", sharedMemorySize);
}

