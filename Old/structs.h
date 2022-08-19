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
	vector<set<int>> territoryDOFsInterior;
	vector<set<int>> territoryDOFsInteriorExt;
	vector<set<int>> territoryDOFsExterior;
	vector<map<int,int>> distanceFromSeedSubdomainMap;

	// Vector of local matrix data structures for each subdomain
	vector<vector<int>> vectorOfIndexPtrs;
	vector<vector<int>> vectorOfNodeNeighbors;
	vector<vector<float>> vectorOfOffdiags;

	// Territory Arrays on CPU
	uint32_t * subdomainOfDOFs;
	uint32_t * distanceFromSeed;
	uint32_t * territoryDOFs;
	uint32_t * distanceFromSeedSubdomain;
	uint32_t * territoryIndexPtr;
	uint32_t * territoryIndexPtrInterior;
	uint32_t * territoryIndexPtrInteriorExt;
	uint32_t * interiorDOFsPerSubdomain;

	// Territory Arrays on GPU
	/* uint32_t * subdomainOfDOFs_d;
	uint32_t * distanceFromSeed_d;
	uint32_t * territoryDOFs_d;
	uint32_t * territoryIndexPtr_d;
	uint32_t * interiorDOFsPerSubdomain_d; */

	// Local Matrix Array on CPU
	uint32_t * indexPtrSubdomain;
	uint32_t * nodeNeighborsSubdomain;
	float * offDiagsSubdomain;
	uint32_t * indexPtrDataShiftSubdomain;
	uint32_t * indexPtrIndexPtrSubdomain;
	
	// Local Matrix Array on GPU
	/* uint32_t * indexPtrSubdomain_d;
	uint32_t * nodeNeighborsSubdomain_d;
	float * offDiagsSubdomain_d;
	uint32_t * indexPtrDataShiftSubdomain_d;
	uint32_t * indexPtrIndexPtrSubdomain_d; */

	// Local rhs and Dinv entries on CPU
	float * rhsLocal;
	float * diagInvLocal;

	// Local rhs and Dinv entries on GPU
	/* float * rhsLocal_d;
	float * diagInvLocal_d; */

	// Shared Memory Allocation
	uint32_t sharedMemorySize;

};

struct meshPartitionForStageNew
{
	// Number of subdomains in this partition
	uint32_t numSubdomains;
	
	// Vectors of sets of seed amd territory DOFs (length of vector is number of subdomains)
	vector<set<int>> seeds;
	vector<set<int>> territoryDOFsInterior;
	vector<set<int>> territoryDOFsInteriorExt;
	vector<set<int>> territoryDOFsExterior;

	// Vector of local matrix data structures for each subdomain
	vector<vector<int>> vectorOfIndexPtrs;
	vector<vector<int>> vectorOfNodeNeighbors;
	vector<vector<float>> vectorOfOffdiags;

	// Territory Arrays on CPU
	uint32_t * subdomainOfDOFs;
	uint32_t * territoryDOFs;
	uint32_t * territoryIndexPtr;
	uint32_t * territoryIndexPtrInterior;
	uint32_t * territoryIndexPtrInteriorExt;

	// Minimum/Maximum iterations for all interior/ext pts to undergo iterations
	uint32_t * maximumIterations;

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

struct meshPartitionForStageDeviceNew
{
	// Number of subdomains in this partition
	uint32_t numSubdomains;
	
	// Territory Arrays on GPU
	uint32_t * subdomainOfDOFs_d;
	uint32_t * territoryDOFs_d;
	uint32_t * territoryIndexPtr_d;
	uint32_t * territoryIndexPtrInterior_d;
	uint32_t * territoryIndexPtrInteriorExt_d;
	
	// Minimum/Maximum iterations for all interior/ext pts to undergo iterations
	uint32_t * maximumIterations_d;

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


struct meshPartitionForStageDevice
{
	// Number of subdomains in this partition
	uint32_t numSubdomains;
	
	// Territory Arrays on GPU
	uint32_t * subdomainOfDOFs_d;
	uint32_t * distanceFromSeed_d;
	uint32_t * distanceFromSeedSubdomain_d;
	uint32_t * territoryDOFs_d;
	uint32_t * territoryIndexPtr_d;
	uint32_t * territoryIndexPtrInterior_d;
	uint32_t * territoryIndexPtrInteriorExt_d;
	uint32_t * interiorDOFsPerSubdomain_d;

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
