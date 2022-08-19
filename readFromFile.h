void readSubdomainAndIterationFromFile2(meshPartitionForStage &partition, std::string PARENT_DIRECTORY, uint32_t stageID)
{

	// ESTABLISH NAMING OF FILES BASED ON THE STAGE
	std::string numDOFsInteriorFileNameBase;
	std::string numDOFsToReturnFileNameBase;
	std::string subdomainDOFsFileNameBase;
	std::string iterationDOFsFileNameBase;
	if (stageID == 0) {
		numDOFsInteriorFileNameBase = PARENT_DIRECTORY + "numDofsInteriorPerSubdomain_1";
		numDOFsToReturnFileNameBase = PARENT_DIRECTORY + "numDofsToReturnPerSubdomain_1";
		subdomainDOFsFileNameBase = PARENT_DIRECTORY + "subdomain_1";
		iterationDOFsFileNameBase = PARENT_DIRECTORY + "iteration_1";
	}
	else if (stageID == 1) {
		numDOFsInteriorFileNameBase = PARENT_DIRECTORY + "numDofsInteriorPerSubdomain_2";
		numDOFsToReturnFileNameBase = PARENT_DIRECTORY + "numDofsToReturnPerSubdomain_2";
		subdomainDOFsFileNameBase = PARENT_DIRECTORY + "subdomain_2";
		iterationDOFsFileNameBase = PARENT_DIRECTORY + "iteration_2";
	}
	else if (stageID == 2) {
		numDOFsInteriorFileNameBase = PARENT_DIRECTORY + "numDofsInteriorPerSubdomain_3";
		numDOFsToReturnFileNameBase = PARENT_DIRECTORY + "numDofsToReturnPerSubdomain_3";
		subdomainDOFsFileNameBase = PARENT_DIRECTORY + "subdomain_3";
		iterationDOFsFileNameBase = PARENT_DIRECTORY + "iteration_3";
	}
	else if (stageID == 3) {
		numDOFsInteriorFileNameBase = PARENT_DIRECTORY + "numDofsInteriorPerSubdomain_4";
		numDOFsToReturnFileNameBase = PARENT_DIRECTORY + "numDofsToReturnPerSubdomain_4";
		subdomainDOFsFileNameBase = PARENT_DIRECTORY + "subdomain_4";
		iterationDOFsFileNameBase = PARENT_DIRECTORY + "iteration_4";
	}

	// READ FILES

	// Initialize variable numDOFs
	uint32_t numDOFs;

	// 1: Read number of dofs per subdomain
	partition.numDOFsInteriorPerSubdomain = new uint32_t[partition.numSubdomains];
	std::ifstream numDOFsInteriorPerSubdomainFile(numDOFsInteriorFileNameBase + ".txt");
	for (int i = 0; i < partition.numSubdomains; i++) {
		numDOFsInteriorPerSubdomainFile >> numDOFs;
		partition.numDOFsInteriorPerSubdomain[i] = numDOFs;
		// printf("The number of dofs in subdomain %d is %d\n", i, numDOFs);
	}

	// 2: Read number of dofs to return per subdomain
	partition.numDOFsToReturnPerSubdomain = new uint32_t[partition.numSubdomains];
	std::ifstream numDOFsToReturnPerSubdomainFile(numDOFsToReturnFileNameBase + ".txt");
	for (int i = 0; i < partition.numSubdomains; i++) {
		numDOFsToReturnPerSubdomainFile >> numDOFs;
		partition.numDOFsToReturnPerSubdomain[i] = numDOFs;
		// printf("The number of dofs to return in subdomain %d is %d\n", i, numDOFs);
	}

	// Create index ptr of dofs per subdomain
	uint32_t * indexPtrFile = new uint32_t[partition.numSubdomains+1];
	indexPtrFile[0] = 0;
	for (int i = 0; i < partition.numSubdomains; i++) {
		indexPtrFile[i+1] = indexPtrFile[i] + partition.numDOFsInteriorPerSubdomain[i];
	}
	
	// Fill in vector of DOFs & corresponding Iteration Level  per subdomain
	vector<int> subdomainDOFs;
	vector<int> iterationDOFs;
	std::ifstream subdomainFile(subdomainDOFsFileNameBase + ".txt");	
	std::ifstream iterationFile(iterationDOFsFileNameBase + ".txt");	
	uint32_t dof;
	uint32_t iter;
	for (int i = 0; i < partition.numSubdomains; i++) {
		//printf("Subdomain %d\n", i);
		//printf("jBounds are (%d, %d)\n", indexPtrFile[i], indexPtrFile[i+1]);
		subdomainDOFs.clear();
		iterationDOFs.clear();
		for (int j = indexPtrFile[i]; j < indexPtrFile[i+1]; j++) {
			subdomainFile >> dof;
			subdomainDOFs.push_back(dof);
			iterationFile >> iter;
			iterationDOFs.push_back(iter);
			// printf("DOF %d\n", dof);
			// printf("Iter %d\n", iter);
		}
		partition.territoryDOFsInterior.push_back(subdomainDOFs);
		partition.iterationLevelPerDOF.push_back(iterationDOFs);
	}
}

//////////////////////////////////////////////////////////////

void readSubdomainAndIterationFromFile(meshPartitionForStage &partition, std::string PARENT_DIRECTORY, uint32_t stageID)
{
	// ESTABLISH NAMING OF FILES BASED ON THE STAGE
	std::string numDOFsInteriorFileNameBase;
	std::string numDOFsToReturnFileNameBase;
	std::string subdomainDOFsFileNameBase;
	std::string iterationDOFsFileNameBase;
	if (stageID == 0) {
		numDOFsInteriorFileNameBase = PARENT_DIRECTORY + "numDofsInteriorPerSubdomain_1";
		numDOFsToReturnFileNameBase = PARENT_DIRECTORY + "numDofsToReturnPerSubdomain_1";
		subdomainDOFsFileNameBase = PARENT_DIRECTORY + "subdomain_1_";
		iterationDOFsFileNameBase = PARENT_DIRECTORY + "iteration_1_";
	}
	else if (stageID == 1) {
		numDOFsInteriorFileNameBase = PARENT_DIRECTORY + "numDofsInteriorPerSubdomain_2";
		numDOFsToReturnFileNameBase = PARENT_DIRECTORY + "numDofsToReturnPerSubdomain_2";
		subdomainDOFsFileNameBase = PARENT_DIRECTORY + "subdomain_2_";
		iterationDOFsFileNameBase = PARENT_DIRECTORY + "iteration_2_";
	}
	else if (stageID == 2) {
		numDOFsInteriorFileNameBase = PARENT_DIRECTORY + "numDofsInteriorPerSubdomain_3";
		numDOFsToReturnFileNameBase = PARENT_DIRECTORY + "numDofsToReturnPerSubdomain_3";
		subdomainDOFsFileNameBase = PARENT_DIRECTORY + "subdomain_3_";
		iterationDOFsFileNameBase = PARENT_DIRECTORY + "iteration_3_";
	}
	else if (stageID == 3) {
		numDOFsInteriorFileNameBase = PARENT_DIRECTORY + "numDofsInteriorPerSubdomain_4";
		numDOFsToReturnFileNameBase = PARENT_DIRECTORY + "numDofsToReturnPerSubdomain_4";
		subdomainDOFsFileNameBase = PARENT_DIRECTORY + "subdomain_4_";
		iterationDOFsFileNameBase = PARENT_DIRECTORY + "iteration_4_";
	}

	// READ FILES
	
	// Initialize variable numDOFs
	uint32_t numDOFs;
	
	// 1: Read number of dofs per subdomain
	partition.numDOFsInteriorPerSubdomain = new uint32_t[partition.numSubdomains];
	std::ifstream numDOFsInteriorPerSubdomainFile(numDOFsInteriorFileNameBase + ".txt");
	for (int i = 0; i < partition.numSubdomains; i++) {
		numDOFsInteriorPerSubdomainFile >> numDOFs;
		partition.numDOFsInteriorPerSubdomain[i] = numDOFs;
		printf("The number of dofs in subdomain %d is %d\n", i, numDOFs);
	}

	// 2: Read number of dofs to return per subdomain
	partition.numDOFsToReturnPerSubdomain = new uint32_t[partition.numSubdomains];
	std::ifstream numDOFsToReturnPerSubdomainFile(numDOFsToReturnFileNameBase + ".txt");
	for (int i = 0; i < partition.numSubdomains; i++) {
		numDOFsToReturnPerSubdomainFile >> numDOFs;
		partition.numDOFsToReturnPerSubdomain[i] = numDOFs;
		printf("The number of dofs to return in subdomain %d is %d\n", i, numDOFs);
	}

	// Fill in vector of DOFs & corresponding Iteration Level  per subdomain
	vector<int> subdomainDOFs;
	vector<int> iterationDOFs;
	for (int i = 0; i < partition.numSubdomains; i++) {
	// for (int i = 0; i < 1; i++) {
		// printf("Subdomain %d\n", i);
		subdomainDOFs.clear();
		iterationDOFs.clear();
		std::ifstream subdomainFile(subdomainDOFsFileNameBase + std::to_string(i) + ".txt");	
		std::ifstream iterationFile(iterationDOFsFileNameBase + std::to_string(i) + ".txt");	
		uint32_t dof;
		for (int j = 0; j < partition.numDOFsInteriorPerSubdomain[i]; j++) {
			subdomainFile >> dof;
			subdomainDOFs.push_back(dof);
			// printf("DOF %d\n", dof);
		}
		partition.territoryDOFsInterior.push_back(subdomainDOFs);
		uint32_t iter;
		for (int j = 0; j < partition.numDOFsInteriorPerSubdomain[i]; j++) {
			iterationFile >> iter;
			iterationDOFs.push_back(iter);
			// printf("Iter %d\n", iter);
		}
		partition.iterationLevelPerDOF.push_back(iterationDOFs);
	}
}
