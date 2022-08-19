void determineSharedMemoryAllocation(meshPartitionForStage &partition, bool lowerPyramidal = false)
{
	// Determine the appropriate shared memory allocation (sum of twice the subdomain DOFs plus number of nnz entries for offdiags)
	uint32_t sharedMemorySize = 0;
	uint32_t sharedMemorySizeTemp;
	uint32_t numDOFs, nnz, numDOFsInteriorPlus1;
	for (int i = 0; i < partition.numSubdomains; i++) {
		numDOFs = partition.territoryIndexPtr[i+1] - partition.territoryIndexPtr[i];
		nnz = partition.indexPtrDataShiftSubdomain[i+1] - partition.indexPtrDataShiftSubdomain[i];
		numDOFsInteriorPlus1 = partition.interiorDOFsPerSubdomain[i]+1;
		if (lowerPyramidal == false) {
			sharedMemorySizeTemp = 2 * numDOFs * sizeof(float) + 2 * numDOFs * sizeof(float) + numDOFsInteriorPlus1 * sizeof(uint32_t) + nnz * sizeof(uint32_t) + nnz * sizeof(float);
		}
		else {
			sharedMemorySizeTemp = (2 * numDOFs) * sizeof(float) + (2 * numDOFs) * sizeof(float) + nnz * sizeof(float); 
		}
		if (sharedMemorySize < sharedMemorySizeTemp) {
			sharedMemorySize = sharedMemorySizeTemp;
		}
	}
	partition.sharedMemorySize = sharedMemorySize;
	printf("The shared memory size is %d\n", sharedMemorySize);
}

__global__
void stageAdvanceJacobiPerformance(float * topSweptSolutionOut, float * bottomSweptSolutionOut, float * topSweptSolutionIn, float * bottomSweptSolutionIn, uint32_t * iterationLevels, meshPartitionForStageDevice partition, uint32_t numJacobiStepsMin, uint32_t numJacobiSteps, uint32_t numThresholdIters)
{
	extern __shared__ float sharedMemorySolution[];
	
	// 1 - Move solution from global memory to shared memory
	uint32_t idx_lower, idx_upper, numDOFs;
	uint32_t i, g, iterLevel;
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
		// Define iteration level 
    	iterLevel = iterationLevels[g];
		// Fill in the shared memory. Whether the most up to date solution is on left or right segment depends on when we will update it
		if (numJacobiStepsMin % 2 == 0) {
			if (iterLevel % 2 == 1) {
				du0[i] = bottomSweptSolutionIn[g];
				du1[i] = topSweptSolutionIn[g];
			}
			else if (iterLevel % 2 == 0) {
				du0[i] = topSweptSolutionIn[g];
				du1[i] = bottomSweptSolutionIn[g];
			}
		}		
		else {
			if (iterLevel % 2 == 1) {
				du1[i] = bottomSweptSolutionIn[g];
				du0[i] = topSweptSolutionIn[g];
			}
			else if (iterLevel % 2 == 0) {
				du1[i] = topSweptSolutionIn[g];
				du0[i] = bottomSweptSolutionIn[g];
			}
		}
		topSweptSolutionOut[g] = topSweptSolutionIn[g];
		bottomSweptSolutionOut[g] = bottomSweptSolutionIn[g];
	}
	__syncthreads();
	// Pointer to matrix and to local rhs and diagInv
	uint32_t shiftIndexPtr = partition.indexPtrIndexPtrSubdomain_d[blockIdx.x];
	uint32_t shiftData = partition.indexPtrDataShiftSubdomain_d[blockIdx.x];
	uint32_t * localIndexPtr = partition.indexPtrSubdomain_d + shiftIndexPtr;
	uint32_t * localNodeNeighbors = partition.nodeNeighborsSubdomain_d + shiftData;
	float * localOffDiags = partition.offDiagsSubdomain_d + shiftData;
	float * localRhs = partition.rhsLocal_d + idx_lower;
	float * localDiagInv = partition.diagInvLocal_d + idx_lower;
	// __syncthreads();
	// Define nnz
	uint32_t nnz = partition.indexPtrDataShiftSubdomain_d[blockIdx.x+1] - partition.indexPtrDataShiftSubdomain_d[blockIdx.x];
	uint32_t numDOFsInterior = partition.interiorDOFsPerSubdomain_d[blockIdx.x];
	// __syncthreads();
	// Define shared pointers
	float * sharedDiagInv = du1 + numDOFs;
	float * sharedRhs = sharedDiagInv + numDOFs;
	uint32_t * sharedIndexPtr = (uint32_t*)&sharedRhs[numDOFs];
	uint32_t * sharedNodeNeighbors =  (uint32_t*)&sharedIndexPtr[numDOFsInterior+1];
	float * sharedOffDiags =  (float*)&sharedNodeNeighbors[nnz];
	// __syncthreads();
	// Fill in shared memory pointers
	for (int i = threadIdx.x; i < nnz; i+= blockDim.x) {
		if (i < idx_upper - idx_lower) {
			sharedDiagInv[i] = localDiagInv[i];
			sharedRhs[i] = localRhs[i];
		}
		if (i < numDOFsInterior+1) {
			sharedIndexPtr[i] = localIndexPtr[i];
		}
		sharedOffDiags[i] = localOffDiags[i];
		sharedNodeNeighbors[i] = localNodeNeighbors[i];
	}
	__syncthreads();

	// 2 - Perform updates within shared memory
	uint32_t dist;
	uint32_t subdomain;
	uint32_t n, ni;
	uint32_t myGroundLevel;
	i = threadIdx.x;
	// __syncthreads();
	if (i < partition.interiorDOFsPerSubdomain_d[blockIdx.x]) {
		g = partition.territoryDOFs_d[idx_lower + i];
		dist = partition.distanceFromSeed_d[g];
		myGroundLevel = iterationLevels[g];
		for (int iter = numJacobiStepsMin; iter < numJacobiSteps; iter++) {
			if ((int)partition.distanceFromSeed_d[g] + iter < numThresholdIters && myGroundLevel <= iter) {
				iterationLevels[g] += 1;
				/* // Jacobi with global data structures
				du1[i] = matrix.rhs_d[g]; // global
				for (int k = localIndexPtr[i]; k < localIndexPtr[i+1]; k++) { // global
					uint32_t n = localNodeNeighbors[k]; // global
					du1[i] -= localOffDiags[k] * du0[n]; // global
				} 
				du1[i] *= matrix.diagInv_d[g]; // global */
				// Jacobi with shared data structures
				du1[i] = sharedRhs[i]; // shared
				for (int k = sharedIndexPtr[i]; k < sharedIndexPtr[i+1]; k++) { // shared
					uint32_t n = sharedNodeNeighbors[k]; // shared
					du1[i] -= sharedOffDiags[k] * du0[n]; // shared
					/*if (blockIdx.x == 224 && threadIdx.x == 6) {
						printf("Iteration %d: Thread %d acts on local dof %d with local neighbor %d with decrement %f times %f for final val %f\n", iter, threadIdx.x, i, n, sharedOffDiags[k], du0[n], du1[i]);
					}*/
				} 
				du1[i] *= sharedDiagInv[i]; // shared 
				/*if (blockIdx.x == 224 && threadIdx.x == 6) {
					printf("Iteration %d: Thread %d acts on local dof %d with division by %f for final val %f\n", iter, threadIdx.x, i, sharedDiagInv[i], du1[i]);
				}*/
			} 	
			__syncthreads();
			float * tmp; tmp = du0; du0 = du1; du1 = tmp;
		}
	}
	__syncthreads();
/*
	if (blockIdx.x == 0) {
		if (i < idx_upper - idx_lower) {
			uint32_t globalDOF = partition.territoryDOFs_d[i];
			uint32_t dist = partition.distanceFromSeed_d[g];
			// printf("Subdomain 0: i = %d, g = %d, dist = %d\n", i, g, dist);
		}
	}
*/
	// 3 - Return solution from shared memory to global memory
	// __syncthreads();
	i = threadIdx.x;
	// __syncthreads();
	if (i < partition.interiorDOFsPerSubdomain_d[blockIdx.x]) {
		// printf("interior DOFs[%d] = %d\n", blockIdx.x, partition.interiorDOFsPerSubdomain_d[blockIdx.x]);
		g = partition.territoryDOFs_d[idx_lower + i];
		dist = partition.distanceFromSeed_d[g];
		myGroundLevel = iterationLevels[g];
		if (numJacobiSteps == numThresholdIters) {
			if (dist % 2 == 0) {
				topSweptSolutionOut[g] = du0[i];
				// printf("%f going into topSweptSolution[%d]\n", du0[i], g);
				bottomSweptSolutionOut[g] = du1[i];
				/* if (blockIdx.x == 224) {
					printf("Above: Block %d: Thread %d transfer to global %d of value %f, %f\n", blockIdx.x, threadIdx.x, g, topSweptSolution[g], bottomSweptSolution[g]);
				} */
			}
			else if (dist % 2 == 1) {
				topSweptSolutionOut[g] = du1[i];
				bottomSweptSolutionOut[g] = du0[i];
				// printf("Block %d: Thread %d transfer to global %d\n", blockIdx.x, threadIdx.x, g);
				/* if (blockIdx.x == 224) {
					printf("Above: Block %d: Thread %d transfer to global %d of value %f, %f\n", blockIdx.x, threadIdx.x, g, topSweptSolution[g], bottomSweptSolution[g]);
				} */
			}
		}
		// do this if in final stage - since all dofs are updated by last iteration, the up to date sol resides in du0 container
		else {
			topSweptSolutionOut[g] = du0[i];
			bottomSweptSolutionOut[g] = du1[i];
		}
		// __syncthreads();
	}
	__syncthreads();
}


__global__
void stageAdvanceJacobiPerformanceLowerPyramidal(float * topSweptSolution, float * bottomSweptSolution, uint32_t * iterationLevels, meshPartitionForStageDevice partition, uint32_t numExpansionSteps, uint32_t numThresholdIters, linearSystemDevice matrix)
{
	extern __shared__ float sharedMemorySolution[];

	// 1 - Move solution from global memory to shared memory
	uint32_t idx_lower, idx_upper, numDOFs;
	uint32_t i, g, iterLevel;
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
		// Define iteration level 
    	iterLevel = iterationLevels[g];
		// Fill in the shared memory. Whether the most up to date solution is on left or right segment depends on when we will update it
		if (iterLevel % 2 == 1) {
			du0[i] = bottomSweptSolution[g];
			du1[i] = topSweptSolution[g];
		}
		else if (iterLevel % 2 == 0) {
			du0[i] = topSweptSolution[g];
			du1[i] = bottomSweptSolution[g];
		}
	}
	__syncthreads();
	// Pointer to matrix
	uint32_t shiftIndexPtr = partition.indexPtrIndexPtrSubdomain_d[blockIdx.x];
	uint32_t shiftData = partition.indexPtrDataShiftSubdomain_d[blockIdx.x];
	uint32_t * localIndexPtr = partition.indexPtrSubdomain_d + shiftIndexPtr;
	uint32_t * localNodeNeighbors = partition.nodeNeighborsSubdomain_d + shiftData;
	float * localOffDiags = partition.offDiagsSubdomain_d + shiftData;
	float * localRhs = partition.rhsLocal_d + idx_lower;
	float * localDiagInv = partition.diagInvLocal_d + idx_lower;
	// Put localOffDiags into shared memory container
	uint32_t nnz = partition.indexPtrDataShiftSubdomain_d[blockIdx.x+1] - partition.indexPtrDataShiftSubdomain_d[blockIdx.x];
	float * sharedDiagInv = du1 + numDOFs;
	float * sharedRhs = sharedDiagInv + numDOFs;
	float * sharedOffDiags = sharedRhs + numDOFs;
	// uint32_t * sharedNodeNeighbors = (uint32_t*)&sharedMemorySolution[2*numDOFs+nnz];
	for (int i = threadIdx.x; i < nnz; i+= blockDim.x) {
		sharedOffDiags[i] = localOffDiags[i];
		if (i < idx_upper - idx_lower) {
			sharedDiagInv[i] = localDiagInv[i];
			sharedRhs[i] = localRhs[i];
		}
	}
	__syncthreads();

	// 2 - Perform updates within shared memory
	uint32_t dist;
	uint32_t subdomain;
	uint32_t n, ni;
	uint32_t myGroundLevel;
	for (int iter = 0; iter < numExpansionSteps; iter++) {
		for (int i = threadIdx.x; i < partition.interiorDOFsPerSubdomain_d[blockIdx.x]; i+= blockDim.x) {
			g = partition.territoryDOFs_d[idx_lower + i];
			dist = partition.distanceFromSeed_d[g];
			myGroundLevel = iterationLevels[g];
			if ((int)partition.distanceFromSeed_d[g] + iter < numThresholdIters && myGroundLevel <= iter) {
				iterationLevels[g] += 1;
				// Jacobi with local data structures
				du1[i] = matrix.rhs_d[g];
				// du1[i] = sharedRhs[g];
				for (int k = localIndexPtr[i]; k < localIndexPtr[i+1]; k++) {
					uint32_t n = localNodeNeighbors[k];
					// uint32_t n = sharedNodeNeighbors[k];
					// uint32_t ng = partition.territoryDOFs_d[idx_lower + n];
					du1[i] -= localOffDiags[k] * du0[n];
					// du1[i] -= sharedOffDiags[k] * du0[n];
				} 
				du1[i] *= matrix.diagInv_d[g]; 
				// du1[i] *= sharedDiagInv[g]; 
			} 
		}
		__syncthreads();
		float * tmp; tmp = du0; du0 = du1; du1 = tmp;
	}
	__syncthreads();

	// 3 - Return solution from shared memory to global memory
	for (int i = threadIdx.x; i < partition.interiorDOFsPerSubdomain_d[blockIdx.x]; i+= blockDim.x) {
		g = partition.territoryDOFs_d[idx_lower + i];
		dist = partition.distanceFromSeed_d[g];
		subdomain = partition.subdomainOfDOFs_d[g];
		// if (numExpansionSteps == numThresholdIters) {
			if (dist % 2 == 0) {
				topSweptSolution[g] = du0[i];
				// printf("%f going into topSweptSolution[%d]\n", du0[i], g);
				bottomSweptSolution[g] = du1[i];
			}
			else if (dist % 2 == 1) {
				topSweptSolution[g] = du1[i];
				// printf("%f going into topSweptSolution[%d]\n", du0[i], g);
				bottomSweptSolution[g] = du0[i];
			}
		// }
		// do this if in final stage - since all dofs are updated by last iteration, the up to date sol resides in du0 container
		//else {
		//	topSweptSolution[g] = du0[i];
		//	bottomSweptSolution[g] = du1[i];
		//}
		__syncthreads();
	}
	__syncthreads();
}
