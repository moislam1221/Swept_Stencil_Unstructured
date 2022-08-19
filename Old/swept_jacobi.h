__global__
void upperPyramidalAdvanceJacobi(float * solution, uint32_t * territoryDOFs, uint32_t * territoryIndexPtr, uint32_t * iterationLevels, uint32_t * distanceFromSeed, uint32_t * indexPtr, uint32_t * nodeNeighbors, uint32_t Ndofs, uint32_t numExpansionSteps, uint32_t shift, uint32_t N, float * rhs, float dx, float * offdiags, float * diagInv)
{
	extern __shared__ float sharedMemorySolution[];

	// 1 - Move solution from global memory to shared memory
	uint32_t idx_lower, idx_upper;
	idx_lower = territoryIndexPtr[blockIdx.x];
	idx_upper = territoryIndexPtr[blockIdx.x + 1];
	uint32_t i = threadIdx.x;
	float * du0 = sharedMemorySolution;
	float * du1 = sharedMemorySolution + shift;
	if (i < idx_upper - idx_lower) {
		du0[i] = solution[i + idx_lower];
		du1[i] = solution[i + idx_lower];
	}
	__syncthreads();
	
	// 2 - Perform updates within shared memory
	uint32_t idx, dof, neighbor, myGroundLevel;
	bool updateDOF, isMember;
	if (i < idx_upper - idx_lower) {
		for (int iter = 0; iter < numExpansionSteps; iter++) {
			idx = threadIdx.x + idx_lower;
			dof = territoryDOFs[idx];
			myGroundLevel = iterationLevels[dof];
			updateDOF = true;
			for (int j = indexPtr[dof]; j < indexPtr[dof+1]; j++) {
				neighbor = nodeNeighbors[j];
				// if neighbor is not a member of all dofs in the set, duplicate DOF IS false
				isMember = false;
				for (int k = idx_lower; k < idx_upper; k++) {
					if (neighbor == territoryDOFs[k] && iterationLevels[neighbor] >= myGroundLevel) {
						isMember = true;
					}
				}
				if (isMember == false) {
					updateDOF = false;
				}
			}
			__syncthreads();
			if (updateDOF == true) {
				// if ((int)myGroundLevel < (iter + 1 - (int)distanceFromSeed[dof]) && distanceFromSeed[dof] != UINT32_MAX) {
					// Simple update
					du1[i] = du0[i] + 1;
					uint32_t gID = territoryDOFs[idx_lower + i];
					printf("Iter %d: Subdomains %d: Update by thread/localID %d on global dof %d (%f -> %f)\n", iter, blockIdx.x, threadIdx.x, gID, du0[i], du1[i]);  
					uint32_t gID = territoryDOFs[idx_lower + i];
					// Jacobi global memory update
					float jac = rhs[gID];
					// printrhs[gID];
					for (int k = indexPtr[gID]; k < indexPtr[gID+1]; k++) {
						uint32_t adjNode = nodeNeighbors[k];
						//if (iterationLevels[neighbor] > myGroundLevel) {
						//	jac += offdiags[adjNode] * du1[adjNode];
						//}
						//else {
							jac -= offdiags[adjNode] * du0[adjNode];
							if (i == 0) {
								printf("gID = %d, contribution from neighbor %d is %.15f\n", gID, adjNode, -diagInv[gID] * offdiags[adjNode] * du0[adjNode]);
							}
						//}
					}
					jac = diagInv[gID] * jac;
					du1[i] = jac;
					printf("Iter %d: Subdomains %d: Update by thread/localID %d on global dof %d (%f -> %f)\n", iter, blockIdx.x, threadIdx.x, gID, du0[i], du1[i]);  
					iterationLevels[dof] += 1; 
					// printf("In iter %d, updating dof %d\n", iter, dof);
				// }
			}
			else {
				 du1[i] = du0[i];
			} 
			__syncthreads();
			float * tmp = du0; du0 = du1; du1 = tmp;
			__syncthreads();
		}
	}
	__syncthreads();

	// 3 - Return solution from shared memory to global memory
	// for (int i = threadIdx.x; i < idx_upper - idx_lower; i += blockDim.x) {
	if (i < idx_upper - idx_lower) {
		solution[i + idx_lower] = du0[i];
		printf("Solution[%d] = %f\n", i, solution[i]); 
	}
	__syncthreads();

}

void advanceFunctionJacobi(float * solution_d, uint32_t * iterationLevel_d, meshPartitionForStage &partition, matrixInfo matrix, uint32_t numExpansionSteps, float * rhs_d, uint32_t N)
{
	// Call function to reorder the solution vector in preparation for transfer to shared memory
	uint32_t numElemsExpanded = partition.territoryIndexPtrExpanded[partition.numSubdomains];
    float * solutionBySubdomain_d;
    cudaMalloc(&solutionBySubdomain_d, sizeof(float) * numElemsExpanded);
	uint32_t threadsPerBlock = 256;
	uint32_t numBlocks = ceil((float)numElemsExpanded / threadsPerBlock);

	// Initialize GPU variables
	// Stage Partition Data
	uint32_t numSubdomains = partition.numSubdomains;
	uint32_t numElems = partition.territoryIndexPtr[numSubdomains];
    uint32_t * distanceFromSeed_d, * territoryDOFs_d, * territoryDOFsExpanded_d, * territoryIndexPtr_d, * territoryIndexPtrExpanded_d;
	cudaMalloc(&distanceFromSeed_d, sizeof(uint32_t) * matrix.Ndofs);
	cudaMalloc(&territoryDOFs_d, sizeof(uint32_t) * numElems);
	cudaMalloc(&territoryDOFsExpanded_d, sizeof(uint32_t) * numElemsExpanded);
	cudaMalloc(&territoryIndexPtr_d, sizeof(uint32_t) * (numSubdomains+1));
	cudaMalloc(&territoryIndexPtrExpanded_d, sizeof(uint32_t) * (numSubdomains+1));
	cudaMemcpy(distanceFromSeed_d, partition.distanceFromSeed, sizeof(uint32_t) * matrix.Ndofs, cudaMemcpyHostToDevice);
	cudaMemcpy(territoryDOFs_d, partition.territoryDOFs, sizeof(uint32_t) * numElems, cudaMemcpyHostToDevice);	
	cudaMemcpy(territoryDOFsExpanded_d, partition.territoryDOFsExpanded, sizeof(uint32_t) * numElemsExpanded, cudaMemcpyHostToDevice);	
	cudaMemcpy(territoryIndexPtr_d, partition.territoryIndexPtr, sizeof(uint32_t) * (numSubdomains+1), cudaMemcpyHostToDevice);
	cudaMemcpy(territoryIndexPtrExpanded_d, partition.territoryIndexPtrExpanded, sizeof(uint32_t) * (numSubdomains+1), cudaMemcpyHostToDevice);
	// Matrix Data
    uint32_t *indexPtr_d, *nodeNeighbors_d;
	float  *offdiags_d, *diagInv_d;
	cudaMalloc(&indexPtr_d, sizeof(uint32_t) * (matrix.Ndofs+1));
	cudaMalloc(&nodeNeighbors_d, sizeof(uint32_t) * matrix.numEntries);
	cudaMalloc(&offdiags_d, sizeof(float) * matrix.numEntries);
	cudaMalloc(&diagInv_d, sizeof(float) * matrix.Ndofs);
	cudaMemcpy(indexPtr_d, matrix.indexPtr, sizeof(uint32_t) * (matrix.Ndofs+1), cudaMemcpyHostToDevice);
	cudaMemcpy(nodeNeighbors_d, matrix.nodeNeighbors, sizeof(uint32_t) * matrix.numEntries, cudaMemcpyHostToDevice);
	cudaMemcpy(offdiags_d, matrix.offdiags, sizeof(float) * matrix.numEntries, cudaMemcpyHostToDevice);
	cudaMemcpy(diagInv_d, matrix.diagInv, sizeof(float) * matrix.Ndofs, cudaMemcpyHostToDevice);
	printDeviceSolutionFloat(rhs_d, N*N, N);

	// Perform the update
	// 1 - Create the solution vector partitioned between different blocks sol = [ block 1 | block 2 | block 3 | etc. ] 
	printf("Step 1\n");
	cudaDeviceSynchronize();
	orderSolutionVectorBySubdomain<<<numBlocks, threadsPerBlock>>>(solutionBySubdomain_d, solution_d, territoryDOFsExpanded_d, numElemsExpanded);
	cudaDeviceSynchronize();

	// 2 - Perform Jacobi updates in shared memory 
	printf("Step 2\n");
	numBlocks = numSubdomains;
	uint32_t maxNumElemsPerSubdomain = 0;
	for (int i = 0; i < numSubdomains; i++) {
		uint32_t maxNumElemsPerSubdomainNext = partition.territoryIndexPtrExpanded[i+1] - partition.territoryIndexPtrExpanded[i];
		if (maxNumElemsPerSubdomainNext > maxNumElemsPerSubdomain) {
			maxNumElemsPerSubdomain = maxNumElemsPerSubdomainNext;
		}
	}
	printf("The max number of elems per subdomains is %d\n", maxNumElemsPerSubdomain);
	float dx = 1.0/(N+1);
	upperPyramidalAdvanceJacobi<<<numBlocks, threadsPerBlock, 2 * sizeof(float) * matrix.Ndofs/4>>>(solutionBySubdomain_d, territoryDOFsExpanded_d, territoryIndexPtrExpanded_d, iterationLevel_d, distanceFromSeed_d, indexPtr_d, nodeNeighbors_d, matrix.Ndofs, numExpansionSteps, maxNumElemsPerSubdomain, N, rhs_d, dx, offdiags_d, diagInv_d);
	cudaDeviceSynchronize();
	printDeviceSolutionFloat(solutionBySubdomain_d, 36, 6);

	// 3 - Transfer solution vector per subdomain into original solution vector 
	printf("Step 3\n");
	undoOrderSolutionVectorBySubdomain<<<numBlocks, threadsPerBlock>>>(solution_d, solutionBySubdomain_d, territoryDOFs_d, territoryIndexPtr_d, territoryDOFsExpanded_d, territoryIndexPtrExpanded_d, matrix.Ndofs);
	cudaDeviceSynchronize();

	// Free up the allocated GPU memory
	cudaFree(distanceFromSeed_d);
	cudaFree(territoryDOFs_d);
	cudaFree(territoryDOFsExpanded_d);
	cudaFree(territoryIndexPtr_d);
	cudaFree(territoryIndexPtrExpanded_d);
	cudaFree(indexPtr_d);
	cudaFree(nodeNeighbors_d);

}
