
void initializeToMaxInt(uint32_t * du, uint32_t Ndofs) 
{
    for (int i = 0; i < Ndofs; i++) {
        du[i] = UINT32_MAX;
    }
}


void initializeToZerosInt(uint32_t * du, uint32_t Ndofs) 
{
    for (int i = 0; i < Ndofs; i++) {
        du[i] = 0;
    }
}

void initializeToZeros(float * du, uint32_t Ndofs) 
{
    for (int i = 0; i < Ndofs; i++) {
        du[i] = 0.0;
    }
}

void initializeToOnes(float * du, uint32_t Ndofs) 
{
    for (int i = 0; i < Ndofs; i++) {
        du[i] = 1.0;
    }
}

void printDeviceSolutionFloat(float * solution_d, uint32_t Ndofs, uint32_t N)
{
	float * solution = new float[Ndofs];
	cudaMemcpy(solution, solution_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%f ", solution[dof]);
		}
		printf("\n");
	}	
}


void printDeviceSolutionInt(uint32_t * solution_d, uint32_t Ndofs, uint32_t N)
{
	uint32_t * solution = new uint32_t[Ndofs];
	cudaMemcpy(solution, solution_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%d ", solution[dof]);
		}
		printf("\n");
	}	
}

void printHostSolutionInt(uint32_t * solution, uint32_t Ndofs, uint32_t N)
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%d ", solution[dof]);
		}
		printf("\n");
	}	
}

void printHostSolutionFloat(float * solution, uint32_t Ndofs, uint32_t N)
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%f ", solution[dof]);
		}
		printf("\n");
	}	
}
////////////////////////////

void initializeMatrixHost(matrixInfo &matrix)
{
	matrix.indexPtr = new uint32_t[matrix.Ndofs+1];
	matrix.nodeNeighbors = new uint32_t[matrix.numEntries];
	matrix.offdiags = new float[matrix.numEntries];
	matrix.diagInv = new float[matrix.Ndofs];
}

void allocateMatrixDevice(matrixInfo &matrix)
{
	cudaMalloc(&matrix.indexPtr_d, sizeof(uint32_t) * (matrix.Ndofs+1));
	cudaMalloc(&matrix.nodeNeighbors_d, sizeof(uint32_t) * matrix.numEntries+1);
	// cudaMalloc(&matrix.matrixData_d, sizeof(float) * matrix.numEntries);
}

void copyMatrixDevice(matrixInfo &matrix)
{
	cudaMemcpy(matrix.indexPtr_d, matrix.indexPtr, sizeof(uint32_t) * (matrix.Ndofs+1), cudaMemcpyHostToDevice);
	cudaMemcpy(matrix.nodeNeighbors_d, matrix.nodeNeighbors, sizeof(uint32_t) * matrix.numEntries, cudaMemcpyHostToDevice);
}

// Create territory data and territory Index Ptr
void createTerritoriesHost(meshPartitionForStage &partition)
{
	uint32_t numSubdomains = partition.numSubdomains;
	partition.territoryIndexPtr = new uint32_t[numSubdomains+1];
	partition.territoryIndexPtrExpanded = new uint32_t[numSubdomains+1];
	partition.territoryIndexPtr[0] = 0;
	partition.territoryIndexPtrExpanded[0] = 0;

	for (int i = 0; i < numSubdomains; i++) {
		partition.territoryIndexPtr[i+1] = partition.territoryIndexPtr[i] + partition.territories[i].size();
		partition.territoryIndexPtrExpanded[i+1] = partition.territoryIndexPtrExpanded[i] + partition.territoriesExpanded[i].size();
	}
	uint32_t numElems = partition.territoryIndexPtr[numSubdomains];
	uint32_t numElemsExpanded = partition.territoryIndexPtrExpanded[numSubdomains];
	partition.territoryDOFs = new uint32_t[numElems];
	partition.territoryDOFsExpanded = new uint32_t[numElemsExpanded];
	uint32_t idx1 = 0;
	uint32_t idx2 = 0;
	for (int i = 0; i < numSubdomains; i++) {
		for (auto elem : partition.territories[i]) {
			partition.territoryDOFs[idx1] = elem;
			// printf("Subdomain %d has element %d\n", i, elem);
			idx1 += 1;
		}
		for (auto elem : partition.territoriesExpanded[i]) {
			partition.territoryDOFsExpanded[idx2] = elem;
			// printf("Expanded Subdomain %d has element %d\n", i, elem);
			idx2 += 1;
		}
	}
}

void allocateTerritoriesDevice(meshPartitionForStage &partition, uint32_t Ndofs)
{
	uint32_t numSubdomains = partition.numSubdomains;
	uint32_t numElems = partition.territoryIndexPtr[numSubdomains];
	uint32_t numElemsExpanded = partition.territoryIndexPtrExpanded[numSubdomains];
    cudaMalloc(&partition.distanceFromSeed_d, sizeof(uint32_t) * Ndofs);
	cudaMalloc(&partition.territoryDOFs_d, sizeof(uint32_t) * numElems);
	cudaMalloc(&partition.territoryDOFsExpanded_d, sizeof(uint32_t) * numElemsExpanded);
	cudaMalloc(&partition.territoryIndexPtr_d, sizeof(uint32_t) * (numSubdomains+1));
	cudaMalloc(&partition.territoryIndexPtrExpanded_d, sizeof(uint32_t) * (numSubdomains+1));
}

void copyTerritoriesDevice(meshPartitionForStage &partition, uint32_t Ndofs) 
{
	uint32_t numSubdomains = partition.numSubdomains;
	uint32_t numElems = partition.territoryIndexPtr[numSubdomains];
	uint32_t numElemsExpanded = partition.territoryIndexPtrExpanded[numSubdomains];
	cudaMemcpy(partition.distanceFromSeed_d, partition.distanceFromSeed, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
	cudaMemcpy(partition.territoryDOFs_d, partition.territoryDOFs, sizeof(uint32_t) * numElems, cudaMemcpyHostToDevice);	
	cudaMemcpy(partition.territoryDOFsExpanded_d, partition.territoryDOFsExpanded, sizeof(uint32_t) * numElemsExpanded, cudaMemcpyHostToDevice);	
	cudaMemcpy(partition.territoryIndexPtr_d, partition.territoryIndexPtr, sizeof(uint32_t) * (numSubdomains+1), cudaMemcpyHostToDevice);
	cudaMemcpy(partition.territoryIndexPtrExpanded_d, partition.territoryIndexPtrExpanded, sizeof(uint32_t) * (numSubdomains+1), cudaMemcpyHostToDevice);
}

