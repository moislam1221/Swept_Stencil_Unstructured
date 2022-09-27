// Initialize the matrix data structures and allocate to host/device

void initializeAndLoadMatrixFromDirectoryDouble(linearSystemDouble &matrix, string directory)
{
	// Number of DOFs
	uint32_t Ndofs = matrix.Ndofs;

	// Load the indexPtr array
	matrix.indexPtr = new uint32_t[Ndofs+1];
	string indexPtrString = "indexPtr.txt";
	std::ifstream indexPtrFile(directory + "/" + indexPtrString);
	for (int i = 0; i < Ndofs+1; i++) {
		indexPtrFile >> matrix.indexPtr[i];
	}

	// Determine the number of nonzeros (nnz) based on last entry of indexPtr
	uint32_t nnz = matrix.indexPtr[Ndofs];
	matrix.numEntries = nnz;
	
	// Load the nodeNeighbors array
	matrix.nodeNeighbors = new uint32_t[nnz];
	string nodeNeighborsString = "nodeNeighbors.txt";
	std::ifstream nodeNeighborFile(directory + "/" + nodeNeighborsString);
	for (int i = 0; i < nnz; i++) {
		nodeNeighborFile >> matrix.nodeNeighbors[i];
	}

	// Load the offDiags array
	matrix.offdiags = new float[nnz];
	string offdiagsString = "offDiags.txt";
	std::ifstream offdiagsFile(directory + "/" + offdiagsString);
	for (int i = 0; i < nnz; i++) {
		offdiagsFile >> matrix.offdiags[i];
	}

	// Load the diagInv array
	matrix.diagInv = new double[Ndofs];
	string diagInvString = "diagInv.txt";
	std::ifstream diagInvFile(directory + "/" + diagInvString);
	for (int i = 0; i < Ndofs; i++) {
		diagInvFile >> matrix.diagInv[i];
	}
	
	// Load the rhs array
	matrix.rhs = new double[Ndofs];
	string rhsString = "rhs.txt";
	std::ifstream rhsFile(directory + "/" + rhsString);
	for (int i = 0; i < Ndofs; i++) {
		rhsFile >> matrix.rhs[i];
	}
}

void initializeAndLoadMatrixFromDirectory(linearSystem &matrix, string directory)
{
	// Number of DOFs
	uint32_t Ndofs = matrix.Ndofs;

	// Load the indexPtr array
	matrix.indexPtr = new uint32_t[Ndofs+1];
	string indexPtrString = "indexPtr.txt";
	std::ifstream indexPtrFile(directory + "/" + indexPtrString);
	for (int i = 0; i < Ndofs+1; i++) {
		indexPtrFile >> matrix.indexPtr[i];
	}

	// Determine the number of nonzeros (nnz) based on last entry of indexPtr
	uint32_t nnz = matrix.indexPtr[Ndofs];
	matrix.numEntries = nnz;
	
	// Load the nodeNeighbors array
	matrix.nodeNeighbors = new uint32_t[nnz];
	string nodeNeighborsString = "nodeNeighbors.txt";
	std::ifstream nodeNeighborFile(directory + "/" + nodeNeighborsString);
	for (int i = 0; i < nnz; i++) {
		nodeNeighborFile >> matrix.nodeNeighbors[i];
	}

	// Load the offDiags array
	matrix.offdiags = new float[nnz];
	string offdiagsString = "offDiags.txt";
	std::ifstream offdiagsFile(directory + "/" + offdiagsString);
	for (int i = 0; i < nnz; i++) {
		offdiagsFile >> matrix.offdiags[i];
	}

	// Load the diagInv array
	matrix.diagInv = new float[Ndofs];
	string diagInvString = "diagInv.txt";
	std::ifstream diagInvFile(directory + "/" + diagInvString);
	for (int i = 0; i < Ndofs; i++) {
		diagInvFile >> matrix.diagInv[i];
	}
	
	// Load the rhs array
	matrix.rhs = new float[Ndofs];
	string rhsString = "rhs.txt";
	std::ifstream rhsFile(directory + "/" + rhsString);
	for (int i = 0; i < Ndofs; i++) {
		rhsFile >> matrix.rhs[i];
	}
}

void initializeAndLoadMatrixFromCSRFiles(linearSystem &matrix)
{
	// Number of DOFs
	uint32_t Ndofs = matrix.Ndofs;

	// Load the indexPtr array
	matrix.indexPtr = new uint32_t[Ndofs+1];
	std::ifstream indexPtrFile("Coarse_Airfoil_Matrix/indexPtr_airfoil_coarse_condensed.txt");
	for (int i = 0; i < Ndofs+1; i++) {
		indexPtrFile >> matrix.indexPtr[i];
	}

	// Determine the number of nonzeros (nnz) based on last entry of indexPtr
	uint32_t nnz = matrix.indexPtr[Ndofs];
	matrix.numEntries = nnz;
	
	// Load the nodeNeighbors array
	matrix.nodeNeighbors = new uint32_t[nnz];
	std::ifstream nodeNeighborFile("Coarse_Airfoil_Matrix/nodeNeighbors_airfoil_coarse_condensed.txt");
	for (int i = 0; i < nnz; i++) {
		nodeNeighborFile >> matrix.nodeNeighbors[i];
	}

	// Load the offDiags array
	matrix.offdiags = new float[nnz];
	std::ifstream offdiagsFile("Coarse_Airfoil_Matrix/offDiags_airfoil_coarse_condensed.txt");
	for (int i = 0; i < nnz; i++) {
		offdiagsFile >> matrix.offdiags[i];
	}

	// Load the diagInv array
	matrix.diagInv = new float[Ndofs];
	std::ifstream diagInvFile("Coarse_Airfoil_Matrix/diagInv_airfoil_coarse_condensed.txt");
	for (int i = 0; i < Ndofs; i++) {
		diagInvFile >> matrix.diagInv[i];
	}
	
	// Load the rhs array
	matrix.rhs = new float[Ndofs];
	std::ifstream rhsFile("Coarse_Airfoil_Matrix/rhs_airfoil_coarse_condensed.txt");
	for (int i = 0; i < Ndofs; i++) {
		rhsFile >> matrix.rhs[i];
	}

}

void fillMatrixNumDOFsEntriesDiagonalLinks(linearSystem &matrix, uint32_t N)
{
	matrix.Ndofs = N*N;
	matrix.numEntries = (N-1)*(N-1)*8 + 4*(N-2)*5 + 4*3;
}

void initializeMatrixHost(linearSystem &matrix, uint32_t N)
{
	matrix.indexPtr = new uint32_t[matrix.Ndofs+1];
	matrix.nodeNeighbors = new uint32_t[matrix.numEntries];
	matrix.offdiags = new float[matrix.numEntries];
	matrix.diagInv = new float[matrix.Ndofs];
	matrix.rhs = new float[matrix.Ndofs];
}

void constructRhs(linearSystem &matrix)
{
	initializeToOnes(matrix.rhs, matrix.Ndofs);
}


// Create local matrix data structures for each subdomain

uint32_t findLocalNeighborIndex(uint32_t globalID, uint32_t * territoryDOFs, uint32_t idx_lower, uint32_t idx_upper)
{
	for (int i = idx_lower; i < idx_upper; i++) {
		if (globalID == territoryDOFs[i]) {
			return i - idx_lower;
		}
	}
	return 0;
}

void constructLocalMatrices(meshPartitionForStage &partition, linearSystem matrix)
{
	// Initalize the local matrix data structures
	vector<int> indexPtrLocal;
	vector<int> nodeNeighborsLocal;
	vector<float> offdiagsLocal;
	uint32_t idx;
	uint32_t n, ni;
	uint32_t lowerBound, upperBound;
	uint32_t interiorDOFsInSubdomain;
	// Fill up the local matrix data structures for all subdomains
	for (int i = 0; i < partition.numSubdomains; i++) {
		indexPtrLocal.clear();
		nodeNeighborsLocal.clear();
		offdiagsLocal.clear();
		indexPtrLocal.push_back(0);
		idx = 0;
		lowerBound = partition.territoryIndexPtr[i];
		upperBound = partition.territoryIndexPtr[i+1];
		interiorDOFsInSubdomain = partition.numDOFsInteriorPerSubdomain[i]; //partition.numDOFsInteriorPerSubdomainVec[i]; // replaced this with vec
		// Fill up the local matrix data structures for this subdomain
		// for (auto dof : partition.territoryDOFsInterior[i]) {
		uint32_t shift = partition.territoryIndexPtr[i];
		for (int i2 = 0; i2 < interiorDOFsInSubdomain; i2++) {
			uint32_t dof = partition.territoryDOFs[i2 + shift]; 
			for (int j = matrix.indexPtr[dof]; j < matrix.indexPtr[dof+1]; j++) {
				n = matrix.nodeNeighbors[j];
				ni = findLocalNeighborIndex(n, partition.territoryDOFs, lowerBound, upperBound); // local version of n within this subdomain
				nodeNeighborsLocal.push_back(ni);
				offdiagsLocal.push_back(matrix.offdiags[j]);
				idx++;			
			}
			indexPtrLocal.push_back(idx);
		}
		// Add these local data structures to the overall array of local matrices
		partition.vectorOfIndexPtrs.push_back(indexPtrLocal);
		partition.vectorOfNodeNeighbors.push_back(nodeNeighborsLocal);
		partition.vectorOfOffdiags.push_back(offdiagsLocal);
	}

}

void allocateLocalMatricesHost(meshPartitionForStage &partition, linearSystem matrix)
{
	uint32_t numElems = partition.territoryIndexPtr[partition.numSubdomains];
	uint32_t numElemsIndexPtr = 0;
	uint32_t numElemsData = 0;
	for (int i = 0; i < partition.numSubdomains; i++) {
		numElemsIndexPtr += partition.vectorOfIndexPtrs[i].size();
		numElemsData += partition.vectorOfIndexPtrs[i].back();
	}

	partition.indexPtrSubdomain = new uint32_t[numElemsIndexPtr];
	partition.nodeNeighborsSubdomain = new uint32_t[numElemsData];
	partition.offDiagsSubdomain = new float[numElemsData];
	partition.rhsLocal = new float[numElems];	
	partition.diagInvLocal = new float[numElems];	
	
	uint32_t idx1 = 0;
	uint32_t idx2 = 0;
	uint32_t idx3 = 0;
	for (int i = 0; i < partition.numSubdomains; i++) {
		for (auto elem : partition.vectorOfIndexPtrs[i]) {
			partition.indexPtrSubdomain[idx1] = elem;
			idx1++;
		}	
		for (auto elem : partition.vectorOfNodeNeighbors[i]) {
			partition.nodeNeighborsSubdomain[idx2] = elem;
			idx2++;
		}	
		for (auto elem : partition.vectorOfOffdiags[i]) {
			partition.offDiagsSubdomain[idx3] = elem;
			idx3++;
		}	
	}
	
	partition.indexPtrIndexPtrSubdomain = new uint32_t[partition.numSubdomains+1];
	partition.indexPtrDataShiftSubdomain = new uint32_t[partition.numSubdomains+1];
	partition.indexPtrIndexPtrSubdomain[0] = 0;
	partition.indexPtrDataShiftSubdomain[0] = 0;
	for (int i = 0; i < partition.numSubdomains; i++) {
		partition.indexPtrIndexPtrSubdomain[i+1] = partition.indexPtrIndexPtrSubdomain[i] + partition.vectorOfIndexPtrs[i].size();
		partition.indexPtrDataShiftSubdomain[i+1] = partition.indexPtrDataShiftSubdomain[i] + partition.vectorOfIndexPtrs[i].back();
	}

	uint32_t globalDOF;	
	for (int i = 0; i < numElems; i++) {
		globalDOF = partition.territoryDOFs[i];
		partition.rhsLocal[i] = matrix.rhs[globalDOF];
		partition.diagInvLocal[i] = matrix.diagInv[globalDOF];
	}

}

void constructLocalMatricesHost(meshPartitionForStage &partition, linearSystem matrix)
{
	// Create a vector of all local matrix data structures on the host
	constructLocalMatrices(partition, matrix);

	// Create data arrays on host and device for all local matrices
	allocateLocalMatricesHost(partition, matrix);
}







