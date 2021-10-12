struct matrixInfo 
{
	uint32_t * idxPtr;
	uint32_t * nodeNeighbors;
	float * matrixData;

	uint32_t Ndofs;
	uint32_t numOffDiags;
	
	uint32_t * idxPtr_d;
	uint32_t * nodeNeighbors_d;
	float * matrixData_d;
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
