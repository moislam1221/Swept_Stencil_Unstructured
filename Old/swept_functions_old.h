void seedsExpandIntoSubdomains(thrust::host_vector<set<int>> &territories, vector<set<int>> seeds, uint32_t * iterationLevel, uint32_t * indexPtr, uint32_t * nodeNeighbors, uint32_t Ndofs, uint32_t numSubdomains, bool expandSet = false)
{

	// Create subdomain of DOFs array to track who owns which dof
	uint32_t * subdomainOfDOFs = new uint32_t[Ndofs];
	for (int i = 0; i < Ndofs; i++) {
		subdomainOfDOFs[i] = UINT32_MAX;
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
		}
	}
	
	// Perform an expansion step for each subdomain, starting from the seed
	int neighbor;
	set<int> setOfNeighborsToAdd;
	for (int iter = 0; iter < 2; iter++) {
		for (int i = 0; i < numSubdomains; i++) {
			for (int level = 0; level < iter+1; level++) {
				for (auto seedDOF : territoriesLevels[i][level]) {
					for (int j = indexPtr[seedDOF]; j < indexPtr[seedDOF+1]; j++) {
						neighbor = nodeNeighbors[j];
						if (subdomainOfDOFs[neighbor] == UINT32_MAX && iterationLevel[neighbor] < iter+1-level) {
							setOfNeighborsToAdd.insert(neighbor);
							subdomainOfDOFs[neighbor] = i;
							printf("Subdomain %d, Level %d: Need to add neighbor = %d\n", i, level, neighbor);
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
/*
	// Add all the neigbors of members to create expanded set
	if (expandSet == true) {
		for (int i = 0; i < numSubdomains; i++) {
			for (auto dof : territories[i]) {
				for (int j = indexPtr[seedDOF]; j < indexPtr[seedDOF+1]; j++) {
					neighbor = nodeNeighbors[j];
					territories[i].insert(neighbor);
				}
			}
		}
	}	
*/				
}

__device__
void expandSubdomainsFromSeeds(uint32_t * subdomainOfDOFs, uint32_t * subdomainOfDOFsCopy, uint32_t * indexPtr, uint32_t * nodeNeighbors, uint32_t * seeds, uint32_t * seedsIndexPtr, uint32_t Ndofs, uint32_t numSubdomains)
{
	// Define variables
	uint32_t j, neighbor;
	bool continueFlag = true;
	
	// Initialize subdomain of dofs by labeling seeds DOFs with corresponding subdomain
	for (int j1 = 0; j1 < numSubdomains; j1++) {
		for (int j2 = seedsIndexPtr[j1]; j2 < seedsIndexPtr[j1+1]; j2++) {
			subdomainOfDOFs[seeds[j2]] = j1;
			subdomainOfDOFsCopy[seeds[j2]] = j1;
		}
	}

	while (continueFlag == true) {
		for (int i = 0; i < Ndofs; i++) {
			if (subdomainOfDOFs[i] != UINT32_MAX) {
				j = subdomainOfDOFs[i];
				for (int k = indexPtr[i]; k < indexPtr[i+1]; k++) {
					neighbor = nodeNeighbors[k];
					if (subdomainOfDOFs[neighbor] == UINT32_MAX) {
						subdomainOfDOFsCopy[neighbor] = j;
					}
					else if (subdomainOfDOFs[neighbor] != j) {
						continueFlag = false;
					}
				}
			}
		}
		for (int i = 0; i < Ndofs; i++) {
			subdomainOfDOFs[i] = subdomainOfDOFsCopy[i];
		}
	}

}


__device__
void orderedSubdomainData(uint32_t * subdomainDOFs, uint32_t * subdomainIndexPtr, uint32_t *  subdomainOfDOFs, uint32_t nDOFs, uint32_t numSubdomains)
{
	// Initialize parameter for subdomain index pointer
	uint32_t idx = 0;
	subdomainIndexPtr[0] = 0;

	// Fill up subdomain DOFs and subdomain index pointer
    // The subdomain DOFs looks like [dofs of subdomain 1, dofs of subdomain 2, etc.]
	for (int j = 0; j < numSubdomains; j++) {
		for (int i = 0; i < nDOFs; i++) {
			if (subdomainOfDOFs[i] == j) {
				subdomainDOFs[idx] = i;
				idx += 1;
			}
		}
		subdomainIndexPtr[j+1] = idx;
	}
}

__global__
void upperPyramidalPartition(uint32_t * dofsOrdered, uint32_t * subdomainIndexPtr, uint32_t * subdomainOfDOFs, uint32_t * subdomainOfDOFsCopy, uint32_t * indexPtr, uint32_t * nodeNeighbors, uint32_t * seeds, uint32_t * seedsIndexPtr, uint32_t Ndofs, uint32_t numSubdomains, uint32_t numExpansionSteps)
{
	// Initialize useful things
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    // Allocate subdomainOfDOFs values to MAXUINT32 
	if (i < Ndofs) {
		subdomainOfDOFs[i] = UINT32_MAX;
		subdomainOfDOFsCopy[i] = UINT32_MAX;
	}

	// Fill up the subdomain of DOFs data structure
	expandSubdomainsFromSeeds(subdomainOfDOFs, subdomainOfDOFsCopy, indexPtr, nodeNeighbors, seeds, seedsIndexPtr, Ndofs, numSubdomains);

    // Order the subdomain data so we have [dofs of subdomain 1, dofs of subdomain 2, etc.] 
	orderedSubdomainData(dofsOrdered, subdomainIndexPtr, subdomainOfDOFs, Ndofs, numSubdomains);
}

__global__
void orderSolutionVectorBySubdomain(float * solutionOrdered, float * solution, uint32_t * dofsOrdered, uint32_t Ndofs)
{
	// Initialize index related parameters
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t index;

	// Create an ordered solution which follows the same ordering of DOFs based on subdomain
	if (i < Ndofs) {
		index = dofsOrdered[i];
		solutionOrdered[i] = solution[index];
	}	
}

__global__
void undoOrderSolutionVectorBySubdomain(float * solution, float * solutionOrdered, uint32_t * dofsOrdered, uint32_t Ndofs)
{
	// Initialize index related parameters
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t index;

	// Create an ordered solution which follows the same ordering of DOFs based on subdomain
	if (i < Ndofs) {
		index = dofsOrdered[i];
		solution[index] = solutionOrdered[i];
	}	
}

__global__
void upperPyramidalAdvance(float * solution, uint32_t * dofs, uint32_t * iterationLevels, uint32_t * subdomainIndexPtr, uint32_t * indexPtr, uint32_t * nodeNeighbors, uint32_t Ndofs)
{
	extern __shared__ float sharedMemorySolution[];

	// 1 - Move solution from global memory to shared memory
	uint32_t idx_lower, idx_upper;
	idx_lower = subdomainIndexPtr[blockIdx.x];
	idx_upper = subdomainIndexPtr[blockIdx.x + 1];
	// for (int i = threadIdx.x; i < idx_upper - idx_lower; i += blockDim.x) {
	uint32_t i = threadIdx.x;
	if (i < idx_upper - idx_lower) {
		sharedMemorySolution[i] = solution[i + idx_lower];
	}
	__syncthreads();

	
	// 2 - Perform updates within shared memory
	uint32_t idx, dof, neighbor, myGroundLevel;
	bool updateDOF, isMember;
	if (i < idx_upper - idx_lower) {
		for (int iter = 0; iter < 2; iter++) {
			idx = threadIdx.x + idx_lower;
			dof = dofs[idx];
			myGroundLevel = iterationLevels[dof];
			updateDOF = true;
			for (int j = indexPtr[dof]; j < indexPtr[dof+1]; j++) {
				neighbor = nodeNeighbors[j];
				// if neighbor is not a member of all dofs in the set, duplicate DOF IS false
				isMember = false;
				
				for (int k = idx_lower; k < idx_upper; k++) {
					if (neighbor == dofs[k]) {
						if (myGroundLevel <= iterationLevels[neighbor]) {
							isMember = true;
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

void upperPyramidal(float * solutionGPU, uint32_t * iterationLevelsGPU, uint32_t * indexPtrGPU, uint32_t * nodeNeighborsGPU, uint32_t * seedsGPU, uint32_t * seedsIndexPtrGPU, uint32_t Ndofs, uint32_t numSubdomains, uint32_t numExpansionSteps)
{
	// Allocate memory for subdomain partition data 
    uint32_t * dofsOrderedGPU, * subdomainIndexPtrGPU, * subdomainOfDOFsGPU, * subdomainOfDOFsGPUCopy;
    cudaMalloc(&dofsOrderedGPU, sizeof(uint32_t) * Ndofs);
    cudaMalloc(&subdomainIndexPtrGPU, sizeof(uint32_t) * (numSubdomains+1));
    cudaMalloc(&subdomainOfDOFsGPU, sizeof(uint32_t) * Ndofs);
    cudaMalloc(&subdomainOfDOFsGPUCopy, sizeof(uint32_t) * Ndofs);

    // Call global function to create upper pyramidal partition
    uint32_t threadsPerBlock = 32;
    uint32_t numBlocks = ceil((float)Ndofs/threadsPerBlock);
	upperPyramidalPartition<<<numBlocks, threadsPerBlock>>>(dofsOrderedGPU, subdomainIndexPtrGPU, subdomainOfDOFsGPU, subdomainOfDOFsGPUCopy, indexPtrGPU, nodeNeighborsGPU, seedsGPU, seedsIndexPtrGPU, Ndofs, numSubdomains, numExpansionSteps);

	// Call function to reorder the solution vector in preparation for transfer to shared memory
    float * solutionOrderedGPU;
    cudaMalloc(&solutionOrderedGPU, sizeof(float) * Ndofs);
	orderSolutionVectorBySubdomain<<<numBlocks, threadsPerBlock>>>(solutionOrderedGPU, solutionGPU, dofsOrderedGPU, Ndofs);
	
    // Call global function to perform the upper pyramidal update
	numBlocks = numSubdomains;
	upperPyramidalAdvance<<<numBlocks, threadsPerBlock, sizeof(float) * (Ndofs/4)>>>(solutionOrderedGPU, dofsOrderedGPU, iterationLevelsGPU, subdomainIndexPtrGPU, indexPtrGPU, nodeNeighborsGPU, Ndofs);

	// Back to global solution ordering	
	undoOrderSolutionVectorBySubdomain<<<numBlocks, threadsPerBlock>>>(solutionGPU, solutionOrderedGPU, dofsOrderedGPU, Ndofs);

}
