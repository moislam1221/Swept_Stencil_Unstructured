/*
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

__device__
uint32_t findLocalIndex(uint32_t globalID, uint32_t * territoryDOFs, uint32_t idx_lower, uint32_t idx_upper)
{
	for (int i = idx_lower; i < idx_upper; i++) {
		if (globalID == territoryDOFs[i]) {
			return i - idx_lower;
		}
	}
	return 0;
}
*/

/*
__global__
void stageAdvanceJacobi(float * topSweptSolution, float * bottomSweptSolution, float * rhs, uint32_t * iterationLevels, meshPartitionForStage partition, matrixInfo matrix, uint32_t numExpansionSteps, uint32_t numThresholdIters, bool verboseFlag)
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
			if (verboseFlag == true) {
				printf("Local dof %d get %f, %f\n", i, du0[i], du1[i]);
			}
		}
		else if (iterLevel % 2 == 0) {
			du0[i] = topSweptSolution[g];
			du1[i] = bottomSweptSolution[g];
			if (verboseFlag == true) {
				printf("Local dof %d get %f, %f\n", i, du0[i], du1[i]);
			}
		}
		// printf("Hello from thread %d and block %d\n", threadIdx.x, blockIdx.x);
	}
	__syncthreads();
	// Pointer to matrix
	uint32_t shiftIndexPtr = partition.indexPtrIndexPtrSubdomain_d[blockIdx.x];
	uint32_t shiftData = partition.indexPtrDataShiftSubdomain_d[blockIdx.x];
	uint32_t * localIndexPtr = partition.indexPtrSubdomain_d + shiftIndexPtr;
	uint32_t * localNodeNeighbors = partition.nodeNeighborsSubdomain_d + shiftData;
	float * localOffDiags = partition.offDiagsSubdomain_d + shiftData;
	// Put localOffDiags into shared memory container
	uint32_t nnz = partition.indexPtrDataShiftSubdomain_d[blockIdx.x+1] - partition.indexPtrDataShiftSubdomain_d[blockIdx.x];
	float * sharedOffDiags = sharedMemorySolution + 2 * numDOFs;
	// uint32_t * sharedNodeNeighbors = (uint32_t*)&sharedMemorySolution[2*numDOFs+nnz];
	for (int i = threadIdx.x; i < nnz; i+= blockDim.x) {
		sharedOffDiags[i] = localOffDiags[i];
		// sharedNodeNeighbors[i] = localNodeNeighbors[i];
	}

	// 2 - Perform updates within shared memory
	uint32_t dist;
	uint32_t subdomain;
	uint32_t n, ni;
	uint32_t myGroundLevel;
	for (int iter = 0; iter < numExpansionSteps; iter++) {
		for (int i = threadIdx.x; i < partition.interiorDOFsPerSubdomain_d[blockIdx.x]; i+= blockDim.x) {
*/			/* if (blockIdx.x == 0 && verboseFlag == true) {
				printf("Subdomain ID: %d, dof: %d, globalDOF: %d, distance: %d\n", blockIdx.x, i, g, dist);
			} */
/*			g = partition.territoryDOFs_d[idx_lower + i];
			dist = partition.distanceFromSeed_d[g];
			myGroundLevel = iterationLevels[g];
			// if ((int)myGroundLevel < (iter + 1 - (int)partition.distanceFromSeed_d[dof]) && partition.distanceFromSeed_d[dof] != UINT32_MAX) {
			// if ((int)partition.distanceFromSeed_d[g] + iter < numExpansionSteps && partition.distanceFromSeed_d[g] != UINT32_MAX) {
			// if ((int)partition.distanceFromSeed_d[g] + iter < numExpansionSteps && myGroundLevel <= iter) {
			if ((int)partition.distanceFromSeed_d[g] + iter < numThresholdIters && myGroundLevel <= iter) {
				/*if (blockIdx.x == 1 && verboseFlag == true) {
					printf("Subdomain ID: %d, dof: %d, globalDOF: %d\n", blockIdx.x, i, g);
				}*/
				iterationLevels[g] += 1;
				// Perform Jacobi update
				// Jacobi with global data structures
				/* 
				du1[i] = rhs[g];
				for (int k = matrix.indexPtr_d[g]; k < matrix.indexPtr_d[g+1]; k++) {
					uint32_t n = matrix.nodeNeighbors_d[k];
					uint32_t ni = findLocalIndex(n, partition.territoryDOFsExpanded_d, idx_lower, idx_upper);
					du1[i] -= matrix.offdiags_d[k] * du0[ni];
					if (blockIdx.x == 0 && verboseFlag == true) {
						// printf("Iteration %d: Updaing local dof %d: global dof %d: neighbor local: %dneighbor term = %f, offdiagterm = %f, neighbor global ID = %d, neighbor local ID = %d\n", iter, i, g, ni, du0[ni], matrix.offdiags_d[k], n, ni);
					} 
				} 
				du1[i] *= matrix.diagInv_d[g];
				*/ 
				// Jacobi with local data structures
				du1[i] = rhs[g];
				for (int k = localIndexPtr[i]; k < localIndexPtr[i+1]; k++) {
					uint32_t n = localNodeNeighbors[k];
					// uint32_t n = sharedNodeNeighbors[k];
					uint32_t ng = partition.territoryDOFs_d[idx_lower + n];
					// du1[i] -= localOffDiags[k] * du0[n];
					du1[i] -= sharedOffDiags[k] * du0[n];
					if (i == 119 && blockIdx.x == 0 && verboseFlag == true) {
						printf("Iteration %d, k = %d: Updating local dof %d: global dof %d: neighbor local: %d, neighbor global: %d, neighbor term = %f, neighbor term alt = %f, offdiagterm = %f, rhs term = %f, diag term = %f\n", iter, k, i, g, n, ng, du0[n], du1[n], localOffDiags[k], rhs[g], matrix.diagInv_d[g]);
					} 
				} 
				du1[i] *= matrix.diagInv_d[g]; 
				if (blockIdx.x == 0 && verboseFlag == true) {
					printf("localIndexPtr[%d] = {%d, %d)\n", i, localIndexPtr[i], localIndexPtr[i+1]);
					printf("Iteration %d: block = %d, dof = %d: du0 = %f -> du1 = %f\n", iter, blockIdx.x, i, du0[i], du1[i]);
				}
			} 
		}
		__syncthreads();
		float * tmp; tmp = du0; du0 = du1; du1 = tmp;
	}

/*
	if (i < 2 * (idx_upper - idx_lower) && blockIdx.x == 1) {
		printf("block ID = %d, sharedMemory[%d] = %f\n", blockIdx.x, i, sharedMemorySolution[i]);
	}
*/

	// 3 - Return solution from shared memory to global memory
	// for (int i = threadIdx.x; i < idx_upper - idx_lower; i += blockDim.x) {
	// if (i < idx_upper - idx_lower) {
	//	solution[i + idx_lower] = sharedMemorySolution[i];
	// }
	if (verboseFlag == true) {
		printf("We are about to transfer data to topSwept array\n");
	}
	for (int i = threadIdx.x; i < partition.interiorDOFsPerSubdomain_d[blockIdx.x]; i+= blockDim.x) {
		if (verboseFlag == true) {
			printf("i is %d out of %d\n", i, partition.interiorDOFsPerSubdomain_d[blockIdx.x]);
		}
		g = partition.territoryDOFs_d[idx_lower + i];
		dist = partition.distanceFromSeed_d[g];
		subdomain = partition.subdomainOfDOFs_d[g];
		if (numExpansionSteps == numThresholdIters) {
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
		}
		// do this if in final stage - since all dofs are updated by last iteration, the up to date sol resides in du0 container
		else {
			topSweptSolution[g] = du0[i];
			bottomSweptSolution[g] = du1[i];
		}
		__syncthreads();
	}
	/*if (blockIdx.x == 0 && threadIdx.x == 0) {
		for (int i = 0; i < 2 * matrix.Ndofs; i++) {
			printf("solutionSwept[%d] = %f\n", i, solutionSwept[i]);
		}
	}*/	

}

__global__
void stageAdvanceJacobiPerformance(float * topSweptSolution, float * bottomSweptSolution, float * rhs, uint32_t * iterationLevels, meshPartitionForStage partition, matrixInfo matrix, uint32_t numExpansionSteps, uint32_t numThresholdIters)
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
	for (int i = threadIdx.x; i < idx_upper - idx_lower; i+= blockDim.x) {
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
	uint32_t numDOFsInterior = partition.interiorDOFsPerSubdomain_d[blockIdx.x];
	// Define shared pointers
	float * sharedDiagInv = du1 + numDOFs;
	float * sharedRhs = sharedDiagInv + numDOFs;
	uint32_t * sharedIndexPtr = (uint32_t*)&sharedRhs[numDOFs];
	uint32_t * sharedNodeNeighbors =  (uint32_t*)&sharedIndexPtr[numDOFsInterior+1];
	float * sharedOffDiags =  (float*)&sharedNodeNeighbors[nnz];
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
	g = partition.territoryDOFs_d[idx_lower + i];
	dist = partition.distanceFromSeed_d[g];
	myGroundLevel = iterationLevels[g];
	if (i < partition.interiorDOFsPerSubdomain_d[blockIdx.x]) {
		for (int iter = 0; iter < numExpansionSteps; iter++) {
			if ((int)partition.distanceFromSeed_d[g] + iter < numThresholdIters && myGroundLevel <= iter) {
				iterationLevels[g] += 1;
				// Jacobi with global data structures
				/* du1[i] = rhs[g]; // global
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
				} 
				du1[i] *= sharedDiagInv[i]; // shared
				// printf("Iteration %d: Thread %d acts on local dof %d at distance %d\n", iter, threadIdx.x, i, dist);
			} 
			__syncthreads();
			float * tmp; tmp = du0; du0 = du1; du1 = tmp;
		}
	}
	__syncthreads();

	// 3 - Return solution from shared memory to global memory
	if (i < partition.interiorDOFsPerSubdomain_d[blockIdx.x]) {
		if (numExpansionSteps == numThresholdIters) {
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
		}
		// do this if in final stage - since all dofs are updated by last iteration, the up to date sol resides in du0 container
		else {
			topSweptSolution[g] = du0[i];
			bottomSweptSolution[g] = du1[i];
		}
		__syncthreads();
	}
	__syncthreads();

}

__global__
void stageAdvanceJacobiPerformanceLowerPyramidal(float * topSweptSolution, float * bottomSweptSolution, float * rhs, uint32_t * iterationLevels, meshPartitionForStage partition, matrixInfo matrix, uint32_t numExpansionSteps, uint32_t numThresholdIters)
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
	for (int i = threadIdx.x; i < idx_upper - idx_lower; i+= blockDim.x) {
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
				// du1[i] = rhs[g];
				du1[i] = sharedRhs[g];
				for (int k = localIndexPtr[i]; k < localIndexPtr[i+1]; k++) {
					uint32_t n = localNodeNeighbors[k];
					// uint32_t n = sharedNodeNeighbors[k];
					uint32_t ng = partition.territoryDOFs_d[idx_lower + n];
					// du1[i] -= localOffDiags[k] * du0[n];
					du1[i] -= sharedOffDiags[k] * du0[n];
				} 
				// du1[i] *= matrix.diagInv_d[g]; 
				du1[i] *= sharedDiagInv[g]; 
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

uint32_t determineSharedMemoryAllocation(meshPartitionForStage partition, bool lowerPyramidal = false)
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
	printf("The shared memory size is %d\n", sharedMemorySize);
	return sharedMemorySize;
}

/*
void advanceFunctionJacobi(float * topSweptSolution_d, float * bottomSweptSolution_d, float * rhs_d, uint32_t * iterationLevel_d, meshPartitionForStage &partition, matrixInfo matrix, uint32_t numExpansionSteps, uint32_t numThresholdIters, bool verboseFlag)
{
	// Define number of blocks and threads per block
	uint32_t threadsPerBlock = 256;
	uint32_t numBlocks = partition.numSubdomains;

	// Determine shared memory allocation
	uint32_t sharedMemoryCount = determineSharedMemoryAllocation(partition);

	// Perform the update
	// (1) Move data to shared memory, (2) perform updates, (3) put most up to date solution in topSweptSolution 
	stageAdvanceJacobi<<<numBlocks, threadsPerBlock, sharedMemoryCount>>>(topSweptSolution_d, bottomSweptSolution_d, rhs_d, iterationLevel_d, partition, matrix, numExpansionSteps, numThresholdIters, verboseFlag);
	cudaDeviceSynchronize();
}
*/

