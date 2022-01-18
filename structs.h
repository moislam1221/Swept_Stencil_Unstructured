struct matrixInfo 
{
	// Matrix data structures
	uint32_t * indexPtr;
	uint32_t * nodeNeighbors;
	float * offdiags;
	float * diagInv;

	// Matrix data structures on the GPU
	uint32_t * indexPtr_d;
	uint32_t * nodeNeighbors_d;
	float * matrixData_d;
	
	// Number of rows and matrix entries
	uint32_t Ndofs;
	uint32_t numEntries;
};


struct linearSystemInfo 
{
	float * rhs;
	float * du0;
	float * du1;
	float * du0_d;
	float * du1_d;
	int32_t Ndofs;
};

struct meshPartitionForStage
{
	// Number of subdomains in this partition
	uint32_t numSubdomains;
	
	// Vectors of sets of seed amd territory DOFs (length of vector is number of subdomains)
	vector<set<int>> seeds;
	vector<set<int>> territories;
	vector<set<int>> territoriesExpanded;

	// Territory Arrays on CPU
	uint32_t * distanceFromSeed;
	uint32_t * territoryIndexPtr;
	uint32_t * territoryIndexPtrExpanded;
	uint32_t * territoryDOFs;
	uint32_t * territoryDOFsExpanded;

	// Territory Arrays on GPU
	uint32_t * distanceFromSeed_d;
	uint32_t * territoryIndexPtr_d;
	uint32_t * territoryIndexPtrExpanded_d;
	uint32_t * territoryDOFs_d;
	uint32_t * territoryDOFsExpanded_d;
};	
	float * offdiagData;
