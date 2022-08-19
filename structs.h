struct linearSystemDouble 
{
	// Matrix data structures
	uint32_t * indexPtr;
	uint32_t * nodeNeighbors;
	float * offdiags;
	double * diagInv;
	double * rhs;

	// Number of rows and matrix entries
	uint32_t Ndofs;
	uint32_t numEntries;
};

struct linearSystem 
{
	// Matrix data structures
	uint32_t * indexPtr;
	uint32_t * nodeNeighbors;
	float * offdiags;
	float * diagInv;
	float * rhs;

	// Number of rows and matrix entries
	uint32_t Ndofs;
	uint32_t numEntries;
};

struct linearSystemDevice
{
	// Matrix data structures on the GPU
	uint32_t * indexPtr_d;
	uint32_t * nodeNeighbors_d;
	float * offdiags_d;
	float * diagInv_d;
	float * rhs_d; 
	
	// Number of rows and matrix entries
	uint32_t Ndofs;
	uint32_t numEntries;
};

struct meshPartitionForStage
{
	// Number of subdomains in this partition
	uint32_t numSubdomains;
	
	// Vectors of sets of seed amd territory DOFs (length of vector is number of subdomains)
	vector<set<int>> seeds;
	vector<vector<int>> territoryDOFsInterior;
	vector<set<int>> territoryDOFsGhost;

	// Vectors of sets of iteration level
	vector<vector<int>> iterationLevelPerDOF;

	// Vector of local matrix data structures for each subdomain
	vector<vector<int>> vectorOfIndexPtrs;
	vector<vector<int>> vectorOfNodeNeighbors;
	vector<vector<float>> vectorOfOffdiags;
	
	// Read from file
	uint32_t * numDOFsInteriorPerSubdomain; 
	uint32_t * numDOFsToReturnPerSubdomain; 

	// Territory Arrays on CPU
	uint32_t * subdomainOfDOFs;
	uint32_t * territoryDOFs;
	uint32_t * territoryIndexPtr;
	uint32_t * iterationLevelDOFs;
	uint32_t * territoryIndexPtrInterior;

	// Local Matrix Array on CPU
	uint32_t * indexPtrSubdomain;
	uint32_t * nodeNeighborsSubdomain;
	float * offDiagsSubdomain;
	uint32_t * indexPtrDataShiftSubdomain;
	uint32_t * indexPtrIndexPtrSubdomain;
	
	// Local rhs and Dinv entries on CPU
	float * rhsLocal;
	float * diagInvLocal;

	// Shared Memory Allocation
	uint32_t sharedMemorySize;

};

struct meshPartitionForStageDevice
{
	// Number of subdomains in this partition
	uint32_t numSubdomains;
	
	// Read from file
	uint32_t * numDOFsInteriorPerSubdomain_d; 
	uint32_t * numDOFsToReturnPerSubdomain_d; 
	
	// Territory Arrays on GPU
	uint32_t * subdomainOfDOFs_d;
	uint32_t * territoryDOFs_d;
	uint32_t * territoryIndexPtr_d;
	uint32_t * territoryIndexPtrInterior_d;
	uint32_t * territoryIndexPtrInteriorExt_d;

	// Minimum/Maximum iterations for all interior/ext pts to undergo iterations
	uint32_t * iterationLevelDOFs_d;

	// Local Matrix Array on GPU
	uint32_t * indexPtrSubdomain_d;
	uint32_t * nodeNeighborsSubdomain_d;
	float * offDiagsSubdomain_d;
	uint32_t * indexPtrDataShiftSubdomain_d;
	uint32_t * indexPtrIndexPtrSubdomain_d;

	// Local rhs and Dinv entries on GPU
	float * rhsLocal_d;
	float * diagInvLocal_d;

};

struct solutionDevice
{
	// Solution Array	
	float * sol_d;

	// Input buffers	
	float * evenSolutionBuffer_d;
	float * oddSolutionBuffer_d;
	uint32_t * iterationLevel_d;

	// Output buffers
	float * evenSolutionBufferOutput_d;
	float * oddSolutionBufferOutput_d;
	uint32_t * iterationLevelOutput_d;
};
	
