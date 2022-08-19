/* Matrix Data Transfer To Device */

// Allocate
void allocateMatrixDevice(linearSystemDevice &matrix_d, linearSystem &matrix)
{
	cudaMalloc(&matrix_d.indexPtr_d, sizeof(uint32_t) * (matrix.Ndofs+1));
	cudaMalloc(&matrix_d.nodeNeighbors_d, sizeof(uint32_t) * matrix.numEntries+1);
	cudaMalloc(&matrix_d.offdiags_d, sizeof(float) * matrix.numEntries);
	cudaMalloc(&matrix_d.diagInv_d, sizeof(float) * matrix.Ndofs);
	cudaMalloc(&matrix_d.rhs_d, sizeof(float) * matrix.Ndofs);
}

// Copy
void copyMatrixDevice(linearSystemDevice &matrix_d, linearSystem &matrix)
{
	matrix_d.Ndofs = matrix.Ndofs;
	matrix_d.numEntries = matrix.numEntries;
	cudaMemcpy(matrix_d.indexPtr_d, matrix.indexPtr, sizeof(uint32_t) * (matrix.Ndofs+1), cudaMemcpyHostToDevice);
	cudaMemcpy(matrix_d.nodeNeighbors_d, matrix.nodeNeighbors, sizeof(uint32_t) * matrix.numEntries, cudaMemcpyHostToDevice);
	cudaMemcpy(matrix_d.offdiags_d, matrix.offdiags, sizeof(uint32_t) * matrix.numEntries, cudaMemcpyHostToDevice);
	cudaMemcpy(matrix_d.diagInv_d, matrix.diagInv, sizeof(uint32_t) * matrix.Ndofs, cudaMemcpyHostToDevice);
	cudaMemcpy(matrix_d.rhs_d, matrix.rhs, sizeof(uint32_t) * matrix.Ndofs, cudaMemcpyHostToDevice);
}

/* Partition Data Transfer To Device */

// Allocate
void allocatePartitionDevice(meshPartitionForStageDevice &partition_d, meshPartitionForStage partition, uint32_t Ndofs)
{
	uint32_t numSubdomains = partition.numSubdomains;
	uint32_t numElems = partition.territoryIndexPtr[numSubdomains];
    cudaMalloc(&partition_d.subdomainOfDOFs_d, sizeof(uint32_t) * Ndofs);
    cudaMalloc(&partition_d.distanceFromSeed_d, sizeof(uint32_t) * Ndofs);
	cudaMalloc(&partition_d.territoryDOFs_d, sizeof(uint32_t) * numElems);
	cudaMalloc(&partition_d.territoryIndexPtr_d, sizeof(uint32_t) * (numSubdomains+1));
	cudaMalloc(&partition_d.interiorDOFsPerSubdomain_d, sizeof(uint32_t) * numSubdomains);

	// Allocate memory on the GPU	
	uint32_t numElemsIndexPtr = 0;
	uint32_t numElemsData = 0;
	for (int i = 0; i < partition.numSubdomains; i++) {
		numElemsIndexPtr += partition.vectorOfIndexPtrs[i].size();
		numElemsData += partition.vectorOfIndexPtrs[i].back();
	}
	cudaMalloc(&partition_d.indexPtrSubdomain_d, sizeof(uint32_t) * numElemsIndexPtr);
	cudaMalloc(&partition_d.nodeNeighborsSubdomain_d, sizeof(uint32_t) * numElemsData);
	cudaMalloc(&partition_d.offDiagsSubdomain_d, sizeof(float) * numElemsData);
	cudaMalloc(&partition_d.indexPtrIndexPtrSubdomain_d, sizeof(float) * (partition.numSubdomains+1));
	cudaMalloc(&partition_d.indexPtrDataShiftSubdomain_d, sizeof(float) * (partition.numSubdomains+1));
	cudaMalloc(&partition_d.rhsLocal_d, sizeof(float) * numElems);
	cudaMalloc(&partition_d.diagInvLocal_d, sizeof(float) * numElems);
}

void allocatePartitionDeviceNew(meshPartitionForStageDeviceNew &partition_d, meshPartitionForStageNew partition, uint32_t Ndofs)
{
	uint32_t numSubdomains = partition.numSubdomains;
	uint32_t numElems = partition.territoryIndexPtr[numSubdomains];
	uint32_t numElemsInteriorExt = partition.territoryIndexPtrInteriorExt[numSubdomains];
    cudaMalloc(&partition_d.subdomainOfDOFs_d, sizeof(uint32_t) * Ndofs);
	cudaMalloc(&partition_d.territoryDOFs_d, sizeof(uint32_t) * numElems);
	cudaMalloc(&partition_d.territoryIndexPtr_d, sizeof(uint32_t) * (numSubdomains+1));
	cudaMalloc(&partition_d.territoryIndexPtrInterior_d, sizeof(uint32_t) * (numSubdomains+1));
	cudaMalloc(&partition_d.territoryIndexPtrInteriorExt_d, sizeof(uint32_t) * (numSubdomains+1));
    cudaMalloc(&partition_d.maximumIterations_d, sizeof(uint32_t) * numElemsInteriorExt);

	// Allocate memory on the GPU	
	uint32_t numElemsIndexPtr = 0;
	uint32_t numElemsData = 0;
	for (int i = 0; i < partition.numSubdomains; i++) {
		numElemsIndexPtr += partition.vectorOfIndexPtrs[i].size();
		numElemsData += partition.vectorOfIndexPtrs[i].back();
	}
	cudaMalloc(&partition_d.indexPtrSubdomain_d, sizeof(uint32_t) * numElemsIndexPtr);
	cudaMalloc(&partition_d.nodeNeighborsSubdomain_d, sizeof(uint32_t) * numElemsData);
	cudaMalloc(&partition_d.offDiagsSubdomain_d, sizeof(float) * numElemsData);
	cudaMalloc(&partition_d.indexPtrIndexPtrSubdomain_d, sizeof(float) * (partition.numSubdomains+1));
	cudaMalloc(&partition_d.indexPtrDataShiftSubdomain_d, sizeof(float) * (partition.numSubdomains+1));
	cudaMalloc(&partition_d.rhsLocal_d, sizeof(float) * numElems);
	cudaMalloc(&partition_d.diagInvLocal_d, sizeof(float) * numElems);
}	

// Copy
void copyPartitionDevice(meshPartitionForStageDevice &partition_d, meshPartitionForStage partition, uint32_t Ndofs)
{
	uint32_t numSubdomains = partition.numSubdomains;
	uint32_t numElems = partition.territoryIndexPtr[numSubdomains];
	partition_d.numSubdomains = numSubdomains;
	cudaMemcpy(partition_d.subdomainOfDOFs_d, partition.subdomainOfDOFs, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.distanceFromSeed_d, partition.distanceFromSeed, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.territoryDOFs_d, partition.territoryDOFs, sizeof(uint32_t) * numElems, cudaMemcpyHostToDevice);	
	cudaMemcpy(partition_d.territoryIndexPtr_d, partition.territoryIndexPtr, sizeof(uint32_t) * (numSubdomains+1), cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.interiorDOFsPerSubdomain_d, partition.interiorDOFsPerSubdomain, sizeof(uint32_t) * numSubdomains, cudaMemcpyHostToDevice);

	// Copy matrix data from CPU to GPU
	uint32_t numElemsIndexPtr = 0;
	uint32_t numElemsData = 0;
	for (int i = 0; i < partition.numSubdomains; i++) {
		numElemsIndexPtr += partition.vectorOfIndexPtrs[i].size();
		numElemsData += partition.vectorOfIndexPtrs[i].back();
	}
	cudaMemcpy(partition_d.indexPtrSubdomain_d, partition.indexPtrSubdomain, sizeof(uint32_t) * numElemsIndexPtr, cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.nodeNeighborsSubdomain_d, partition.nodeNeighborsSubdomain, sizeof(uint32_t) * numElemsData, cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.offDiagsSubdomain_d, partition.offDiagsSubdomain, sizeof(float) * numElemsData, cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.indexPtrIndexPtrSubdomain_d, partition.indexPtrIndexPtrSubdomain, sizeof(uint32_t) * (partition.numSubdomains+1), cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.indexPtrDataShiftSubdomain_d, partition.indexPtrDataShiftSubdomain, sizeof(uint32_t) * (partition.numSubdomains+1), cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.rhsLocal_d, partition.rhsLocal, sizeof(float) * numElems, cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.diagInvLocal_d, partition.diagInvLocal, sizeof(float) * numElems, cudaMemcpyHostToDevice);
}

void copyPartitionDeviceNew(meshPartitionForStageDeviceNew &partition_d, meshPartitionForStageNew partition, uint32_t Ndofs)
{
	uint32_t numSubdomains = partition.numSubdomains;
	uint32_t numElems = partition.territoryIndexPtr[numSubdomains];
	uint32_t numElemsInteriorExt = partition.territoryIndexPtrInteriorExt[numSubdomains];
	partition_d.numSubdomains = numSubdomains;
	cudaMemcpy(partition_d.subdomainOfDOFs_d, partition.subdomainOfDOFs, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.territoryDOFs_d, partition.territoryDOFs, sizeof(uint32_t) * numElems, cudaMemcpyHostToDevice);	
	cudaMemcpy(partition_d.territoryIndexPtr_d, partition.territoryIndexPtr, sizeof(uint32_t) * (numSubdomains+1), cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.territoryIndexPtrInterior_d, partition.territoryIndexPtrInterior, sizeof(uint32_t) * (numSubdomains+1), cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.territoryIndexPtrInteriorExt_d, partition.territoryIndexPtrInteriorExt, sizeof(uint32_t) * (numSubdomains+1), cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.maximumIterations_d, partition.maximumIterations, sizeof(uint32_t) * numElemsInteriorExt, cudaMemcpyHostToDevice);	

	// Copy matrix data from CPU to GPU
	uint32_t numElemsIndexPtr = 0;
	uint32_t numElemsData = 0;
	for (int i = 0; i < partition.numSubdomains; i++) {
		numElemsIndexPtr += partition.vectorOfIndexPtrs[i].size();
		numElemsData += partition.vectorOfIndexPtrs[i].back();
	}
	cudaMemcpy(partition_d.indexPtrSubdomain_d, partition.indexPtrSubdomain, sizeof(uint32_t) * numElemsIndexPtr, cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.nodeNeighborsSubdomain_d, partition.nodeNeighborsSubdomain, sizeof(uint32_t) * numElemsData, cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.offDiagsSubdomain_d, partition.offDiagsSubdomain, sizeof(float) * numElemsData, cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.indexPtrIndexPtrSubdomain_d, partition.indexPtrIndexPtrSubdomain, sizeof(uint32_t) * (partition.numSubdomains+1), cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.indexPtrDataShiftSubdomain_d, partition.indexPtrDataShiftSubdomain, sizeof(uint32_t) * (partition.numSubdomains+1), cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.rhsLocal_d, partition.rhsLocal, sizeof(float) * numElems, cudaMemcpyHostToDevice);
	cudaMemcpy(partition_d.diagInvLocal_d, partition.diagInvLocal, sizeof(float) * numElems, cudaMemcpyHostToDevice);
}












