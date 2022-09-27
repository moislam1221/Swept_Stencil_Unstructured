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
	
    // Seed DOFs (this concept only used for structured case now)
	vector<set<int>> seeds; 
	uint32_t * subdomainOfDOFs; // only used in structured case

	// Set of interior, ghost and overall DOFs
	vector<vector<int>> territoryDOFsInterior;
	vector<set<int>> territoryDOFsGhost;
	uint32_t * territoryDOFs;

	// Vectors of sets of iteration level
	// Vector of interation level
    vector<vector<int>> iterationLevelPerDOF;
	// Flattened data
    uint32_t * iterationLevelDOFs;
	
	// Read from file
	vector<uint32_t> numDOFsInteriorPerSubdomain;
	vector<uint32_t> numDOFsToReturnPerSubdomain; 

	// Territory Arrays on CPU
	uint32_t * territoryIndexPtr;
	uint32_t * territoryIndexPtrInterior;

	// Local Matrix and rhs information on CPU
	// Vector of L+U for each subdomain (needs to be flattened to data below)
	vector<vector<int>> vectorOfIndexPtrs;
	vector<vector<int>> vectorOfNodeNeighbors;
	vector<vector<float>> vectorOfOffdiags;
	// Flattened matrix data
    // L + U
	uint32_t * indexPtrSubdomain;
	uint32_t * nodeNeighborsSubdomain;
	float * offDiagsSubdomain;
	// b
    float * rhsLocal;
	// D^{-1}
    float * diagInvLocal;
	// Index Ptrs to jump between subdomains
	uint32_t * indexPtrDataShiftSubdomain;
	uint32_t * indexPtrIndexPtrSubdomain;
	
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
	uint32_t * iterationLevelDOFs_d; // Minimum/Maximum iterations for all interior/ext pts to undergo iterations
	
	// Territory Arrays on GPU
	uint32_t * subdomainOfDOFs_d;
	uint32_t * territoryDOFs_d;
	uint32_t * territoryIndexPtr_d;
	uint32_t * territoryIndexPtrInterior_d;

	// Local Matrix and rhs information on GPU
    // L + U
	uint32_t * indexPtrSubdomain_d;
	uint32_t * nodeNeighborsSubdomain_d;
	float * offDiagsSubdomain_d;
	// b
    float * rhsLocal_d;
	// D^{-1}
    float * diagInvLocal_d;
	// Index Ptrs to jump between subdomains
	uint32_t * indexPtrDataShiftSubdomain_d;
	uint32_t * indexPtrIndexPtrSubdomain_d;
	
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
	
