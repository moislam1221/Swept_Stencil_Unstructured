void seedsExpandIntoSubdomains(vector<set<int>> &territories, vector<set<int>> seeds, uint32_t * iterationLevel, uint32_t * distanceFromSeed, uint32_t * indexPtr, uint32_t * nodeNeighbors, uint32_t Ndofs, uint32_t numSubdomains, uint32_t numExpansionSteps)
{

	// Create subdomain of DOFs array to track who owns which dof
	uint32_t * subdomainOfDOFs = new uint32_t[Ndofs];
	for (int i = 0; i < Ndofs; i++) {
		subdomainOfDOFs[i] = UINT32_MAX;
		distanceFromSeed[i] = UINT32_MAX;
	}

	// Copy the seeds into the territories vector
	vector<vector<set<int>>> territoriesLevels;
	vector<set<int>> dummyVector;
	for (int i = 0; i < numSubdomains; i++) {
		dummyVector.push_back(seeds[i]);
		territoriesLevels.push_back(dummyVector);
		dummyVector.clear();
		for (auto seedDOF : seeds[i]) {
			subdomainOfDOFs[seedDOF] = i;
			distanceFromSeed[seedDOF] = 0;
		}
	}
	
	// Perform an expansion step for each subdomain, starting from the seed
	int neighbor;
	set<int> setOfNeighborsToAdd;
	for (int iter = 0; iter < numExpansionSteps; iter++) {
		for (int i = 0; i < numSubdomains; i++) {
			for (int level = 0; level < iter+1; level++) {
				for (auto seedDOF : territoriesLevels[i][level]) {
					for (int j = indexPtr[seedDOF]; j < indexPtr[seedDOF+1]; j++) {
						neighbor = nodeNeighbors[j];
						if (subdomainOfDOFs[neighbor] == UINT32_MAX && iterationLevel[neighbor] < iter+1-level) {
							setOfNeighborsToAdd.insert(neighbor);
							subdomainOfDOFs[neighbor] = i;
							distanceFromSeed[neighbor] = level+1;
							// printf("Subdomain %d, Level %d: Need to add neighbor = %d\n", i, level, neighbor);
						}
					}
				}
				if (level == iter) {
					territoriesLevels[i].push_back(setOfNeighborsToAdd);
				}
				else {		
					territoriesLevels[i][level].insert(setOfNeighborsToAdd.begin(), setOfNeighborsToAdd.end());
				}
				setOfNeighborsToAdd.clear();
			}
		}
	}

	// Concatenate all the contents of each subdomain's territories into a single set
	set<int> mergedSetInSubdomain;
	for (int i = 0; i < numSubdomains; i++) {
		mergedSetInSubdomain.clear();
		for (auto seedSet : territoriesLevels[i]) {
			mergedSetInSubdomain.insert(seedSet.begin(), seedSet.end());
		}
		territories.push_back(mergedSetInSubdomain);
	}

}

void expandToHaloRegions(vector<set<int>> &territoriesExpanded, vector<set<int>> &territories, uint32_t * indexPtr, uint32_t * nodeNeighbors, uint32_t numSubdomains)
{
	// Copy territories into territoriesExpanded
	for (int i = 0; i < numSubdomains; i++) {
		territoriesExpanded.push_back(territories[i]);
	}	

	// Add all the neigbors of members to create expanded set
	uint32_t neighbor;
	for (int i = 0; i < numSubdomains; i++) {
		for (auto dof : territories[i]) {
			for (int j = indexPtr[dof]; j < indexPtr[dof+1]; j++) {
				neighbor = nodeNeighbors[j];
				territoriesExpanded[i].insert(neighbor);
				printf("Subdomain %d: Adding DOF %d\n", i, dof);
			}
		}
	}
}

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
void undoOrderSolutionVectorBySubdomain(float * solution, float * solutionBySubdomain, uint32_t * territoryDOFs, uint32_t * territoryIndexPtr, uint32_t * territoryDOFsExpanded, uint32_t * territoryIndexPtrExpanded, uint32_t Ndofs)
{
	// Initialize index related parameters
	uint32_t i = threadIdx.x;
	uint32_t idx, index;
	bool update;
	uint32_t idx_lower_inner = territoryIndexPtr[blockIdx.x];
	uint32_t idx_upper_inner = territoryIndexPtr[blockIdx.x + 1];
	uint32_t idx_lower = territoryIndexPtrExpanded[blockIdx.x];
	uint32_t idx_upper = territoryIndexPtrExpanded[blockIdx.x + 1];

	// Create an ordered solution which follows the same ordering of DOFs based on subdomain
	if (i < idx_upper - idx_lower) {
		idx = i + idx_lower;
		index = territoryDOFsExpanded[idx];
		update = false;
		for (int j = idx_lower_inner; j < idx_upper_inner; j++) {
			if (territoryDOFsExpanded[idx] == territoryDOFs[j]) {
				update = true;
				j = idx_upper_inner; // end the for loop
			}
		}
		if (update == true) {
			// printf("Subdomain %d: Thread %d, Update dof %d\n", blockIdx.x, i, index);
			solution[index] = solutionBySubdomain[idx];
		}
	}
}

__global__
void upperPyramidalAdvance(float * solution, uint32_t * territoryDOFs, uint32_t * territoryIndexPtr, uint32_t * iterationLevels, uint32_t * distanceFromSeed, uint32_t * indexPtr, uint32_t * nodeNeighbors, uint32_t Ndofs, uint32_t numExpansionSteps)
{
	extern __shared__ float sharedMemorySolution[];

	// 1 - Move solution from global memory to shared memory
	uint32_t idx_lower, idx_upper;
	idx_lower = territoryIndexPtr[blockIdx.x];
	idx_upper = territoryIndexPtr[blockIdx.x + 1];
	// for (int i = threadIdx.x; i < idx_upper - idx_lower; i += blockDim.x) {
	uint32_t i = threadIdx.x;
	if (i < idx_upper - idx_lower) {
		sharedMemorySolution[i] = solution[i + idx_lower];
		// printf("Block %d, sharedMemory[%d] corresponding to %d = %f\n", blockIdx.x, i, territoryDOFs[i+idx_lower], sharedMemorySolution[i]);
	}
	__syncthreads();
	
	// 2 - Perform updates within shared memory
	uint32_t idx, dof, neighbor, myGroundLevel;
	bool updateDOF, isMember;
	if (i < idx_upper - idx_lower) {
		for (int iter = 0; iter < numExpansionSteps; iter++) {
			if (i == 0) {
				// printf("Iter = %d\n", iter);
			}
			idx = threadIdx.x + idx_lower;
			dof = territoryDOFs[idx];
			// printf("Subdomain %d: Check dof %d\n", blockIdx.x, dof);
			myGroundLevel = iterationLevels[dof];
			updateDOF = true;
			for (int j = indexPtr[dof]; j < indexPtr[dof+1]; j++) {
				neighbor = nodeNeighbors[j];
				// if neighbor is not a member of all dofs in the set, duplicate DOF IS false
				isMember = false;
				for (int k = idx_lower; k < idx_upper; k++) {
					if (neighbor == territoryDOFs[k]) {
						isMember = true;
					}
				}
				if (isMember == false) {
					updateDOF = false;
					// printf("Here for dof %d and neighbor %d\n", dof, neighbor);
				}
			}
			__syncthreads();
			if (updateDOF == true) {
				if (distanceFromSeed[dof] != UINT32_MAX) {
				// printf("For dof %d: The seed distance is %d and myGroundLevel is %d\n", dof, distanceFromSeed[dof], myGroundLevel);
					// printf("Not updated: In iter %d, updating %d with seed distance %d. The bound was %d and our my ground is %d\n", iter, dof, distanceFromSeed[dof], (int)(iter+1-distanceFromSeed[dof]), myGroundLevel);
				}
				if ((int)myGroundLevel < (iter + 1 - (int)distanceFromSeed[dof]) && distanceFromSeed[dof] != UINT32_MAX) {
					sharedMemorySolution[i] += 1;
					// printf("In iter %d, updating %d with seed distance %d. The bound was %d and our my ground is %d\n", iter, dof, distanceFromSeed[dof], (int)(iter+1-distanceFromSeed[dof]), myGroundLevel);
					// printf("Subdomain %d: Iter %d: Updating dof %d from %f to %f at iteration level %d\n", blockIdx.x, iter, dof, sharedMemorySolution[i]-1, sharedMemorySolution[i], iterationLevels[dof]);
					iterationLevels[dof] += 1;
				}
			} 
		__syncthreads();
		}
	}

	// 3 - Return solution from shared memory to global memory
	// for (int i = threadIdx.x; i < idx_upper - idx_lower; i += blockDim.x) {
	if (i < idx_upper - idx_lower) {
		solution[i + idx_lower] = sharedMemorySolution[i];
	}
	__syncthreads();

}






















__global__
void upperPyramidalAdvanceOld(float * solution, uint32_t * territoryDOFs, uint32_t * territoryIndexPtr, uint32_t * iterationLevels, uint32_t * indexPtr, uint32_t * nodeNeighbors, uint32_t Ndofs, uint32_t numExpansionSteps)
{
	extern __shared__ float sharedMemorySolution[];

	// 1 - Move solution from global memory to shared memory
	uint32_t idx_lower, idx_upper;
	idx_lower = territoryIndexPtr[blockIdx.x];
	idx_upper = territoryIndexPtr[blockIdx.x + 1];
	// for (int i = threadIdx.x; i < idx_upper - idx_lower; i += blockDim.x) {
	uint32_t i = threadIdx.x;
	if (i < idx_upper - idx_lower) {
		sharedMemorySolution[i] = solution[i + idx_lower];
		// printf("Block %d, sharedMemory[%d] corresponding to %d = %f\n", blockIdx.x, i, territoryDOFs[i+idx_lower], sharedMemorySolution[i]);
	}
	__syncthreads();
	
	// 2 - Perform updates within shared memory
	uint32_t idx, dof, neighbor, myGroundLevel;
	bool updateDOF, isMember;
	if (i < idx_upper - idx_lower) {
		for (int iter = 0; iter < numExpansionSteps; iter++) {
			if (i == 0) {
				printf("Iter = %d\n", iter);
			}
			idx = threadIdx.x + idx_lower;
			dof = territoryDOFs[idx];
			myGroundLevel = iterationLevels[dof];
			updateDOF = true;
			for (int j = indexPtr[dof]; j < indexPtr[dof+1]; j++) {
				neighbor = nodeNeighbors[j];
				// if neighbor is not a member of all dofs in the set, duplicate DOF IS false
				isMember = false;
				
				for (int k = idx_lower; k < idx_upper; k++) {
					if (neighbor == territoryDOFs[k]) {
						if (myGroundLevel <= iterationLevels[neighbor]) {
							isMember = true;
							// printf("Iter %d: For dof %d, neighbor %d is valid\n", iter, dof, neighbor);
						}
					}
				}
				if (isMember == false) {
					updateDOF = false;
				}
			}
			__syncthreads();
			if (updateDOF == true) {
				sharedMemorySolution[i] += 1;
				printf("In iter %d, updating %d\n", iter, dof);
				// printf("Subdomain %d: Iter %d: Updating dof %d from %f to %f at iteration level %d\n", blockIdx.x, iter, dof, sharedMemorySolution[i]-1, sharedMemorySolution[i], iterationLevels[dof]);
				iterationLevels[dof] += 1;
			} 
		__syncthreads();
		}
	}

	// 3 - Return solution from shared memory to global memory
	// for (int i = threadIdx.x; i < idx_upper - idx_lower; i += blockDim.x) {
	if (i < idx_upper - idx_lower) {
		solution[i + idx_lower] = sharedMemorySolution[i];
	}
	__syncthreads();

}

void stageAdvance(float * solutionGPU, uint32_t * territoryDOFsGPU, uint32_t * territoryIndexPtrGPU, uint32_t * territoryDOFsExpandedGPU, uint32_t * territoryIndexPtrExpandedGPU, uint32_t * iterationLevelsGPU, uint32_t * distanceFromSeedGPU, uint32_t * indexPtrGPU, uint32_t * nodeNeighborsGPU, uint32_t Ndofs, uint32_t numSubdomains, uint32_t numExpansionSteps, uint32_t numElemsExpanded)
{
	// Call function to reorder the solution vector in preparation for transfer to shared memory
    float * solutionBySubdomainGPU;
    cudaMalloc(&solutionBySubdomainGPU, sizeof(float) * numElemsExpanded);
	uint32_t threadsPerBlock = 32;
	uint32_t numBlocks = ceil((float)numElemsExpanded / threadsPerBlock);

	orderSolutionVectorBySubdomain<<<numBlocks, threadsPerBlock>>>(solutionBySubdomainGPU, solutionGPU, territoryDOFsExpandedGPU, numElemsExpanded);

    // Call global function to perform the upper pyramidal update
	numBlocks = numSubdomains;
	upperPyramidalAdvance<<<numBlocks, threadsPerBlock, 2 * sizeof(float) * (Ndofs/4)>>>(solutionBySubdomainGPU, territoryDOFsExpandedGPU, territoryIndexPtrExpandedGPU, iterationLevelsGPU, distanceFromSeedGPU, indexPtrGPU, nodeNeighborsGPU, Ndofs, numExpansionSteps);
	// upperPyramidalAdvance<<<numBlocks, threadsPerBlock, sizeof(float) * (Ndofs/4)>>>(solutionBySubdomainGPU, territoryDOFsExpandedGPU, territoryIndexPtrExpandedGPU, iterationLevelsGPU, indexPtrGPU, nodeNeighborsGPU, Ndofs, numExpansionSteps);

	// Back to global solution ordering	
	undoOrderSolutionVectorBySubdomain<<<numBlocks, threadsPerBlock>>>(solutionGPU, solutionBySubdomainGPU, territoryDOFsGPU, territoryIndexPtrGPU, territoryDOFsExpandedGPU, territoryIndexPtrExpandedGPU, Ndofs);

}







void upperPyramidalNew(float * solutionGPU, uint32_t * territoryDOFsGPU, uint32_t * territoryIndexPtrGPU, uint32_t * territoryDOFsExpandedGPU, uint32_t * territoryIndexPtrExpandedGPU, uint32_t * iterationLevelsGPU, uint32_t * distanceFromSeedGPU, uint32_t * indexPtrGPU, uint32_t * nodeNeighborsGPU, uint32_t Ndofs, uint32_t numSubdomains, uint32_t numExpansionSteps, uint32_t numElemsExpanded)
{
	// Call function to reorder the solution vector in preparation for transfer to shared memory
    float * solutionBySubdomainGPU;
    cudaMalloc(&solutionBySubdomainGPU, sizeof(float) * numElemsExpanded);
	uint32_t threadsPerBlock = 32;
	uint32_t numBlocks = ceil((float)numElemsExpanded / threadsPerBlock);

	orderSolutionVectorBySubdomain<<<numBlocks, threadsPerBlock>>>(solutionBySubdomainGPU, solutionGPU, territoryDOFsExpandedGPU, numElemsExpanded);

    // Call global function to perform the upper pyramidal update
	numBlocks = numSubdomains;
	upperPyramidalAdvance<<<numBlocks, threadsPerBlock, 2 * sizeof(float) * (Ndofs/4)>>>(solutionBySubdomainGPU, territoryDOFsExpandedGPU, territoryIndexPtrExpandedGPU, iterationLevelsGPU, distanceFromSeedGPU, indexPtrGPU, nodeNeighborsGPU, Ndofs, numExpansionSteps);
	// upperPyramidalAdvance<<<numBlocks, threadsPerBlock, sizeof(float) * (Ndofs/4)>>>(solutionBySubdomainGPU, territoryDOFsExpandedGPU, territoryIndexPtrExpandedGPU, iterationLevelsGPU, indexPtrGPU, nodeNeighborsGPU, Ndofs, numExpansionSteps);

	// Back to global solution ordering	
	undoOrderSolutionVectorBySubdomain<<<numBlocks, threadsPerBlock>>>(solutionGPU, solutionBySubdomainGPU, territoryDOFsGPU, territoryIndexPtrGPU, territoryDOFsExpandedGPU, territoryIndexPtrExpandedGPU, Ndofs);

}

void upperPyramidal(float * solutionGPU, uint32_t * territoryDOFsGPU, uint32_t * territoryIndexPtrGPU, uint32_t * iterationLevelsGPU, uint32_t * indexPtrGPU, uint32_t * nodeNeighborsGPU, uint32_t Ndofs, uint32_t numSubdomains, uint32_t numExpansionSteps)
{
	// Call function to reorder the solution vector in preparation for transfer to shared memory
    float * solutionOrderedGPU;
    cudaMalloc(&solutionOrderedGPU, sizeof(float) * Ndofs);
	uint32_t threadsPerBlock = 32;
	uint32_t numBlocks = ceil((float)Ndofs / threadsPerBlock);

	orderSolutionVectorBySubdomain<<<numBlocks, threadsPerBlock>>>(solutionOrderedGPU, solutionGPU, territoryDOFsGPU, Ndofs);
	
    // Call global function to perform the upper pyramidal update
	numBlocks = numSubdomains;
	// upperPyramidalAdvance<<<numBlocks, threadsPerBlock, sizeof(float) * (Ndofs/4)>>>(solutionOrderedGPU, territoryDOFsGPU, territoryIndexPtrGPU, iterationLevelsGPU, indexPtrGPU, nodeNeighborsGPU, Ndofs);

	// Back to global solution ordering	
	// undoOrderSolutionVectorBySubdomain<<<numBlocks, threadsPerBlock>>>(solutionGPU, solutionOrderedGPU, territoryDOFsGPU, Ndofs);

}
