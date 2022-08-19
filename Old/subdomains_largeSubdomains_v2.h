// Create territories based on seeds and expand for exterior DOFs

void seedsExpandIntoSubdomains(meshPartitionForStage &partition, linearSystem matrix, uint32_t * iterationLevel, uint32_t numExpansionSteps)
{
	// Initialize subdomain of DOFs and distance from seeds data pointers
	partition.distanceFromSeed = new uint32_t[matrix.Ndofs];
	partition.subdomainOfDOFs = new uint32_t[matrix.Ndofs];

	// Load subdomain of DOFs and distance arrays to track who owns which dof and how far they are from seeds (useful info for iterations)
	for (int i = 0; i < matrix.Ndofs; i++) {
		partition.subdomainOfDOFs[i] = UINT32_MAX;
		partition.distanceFromSeed[i] = UINT32_MAX;
	}

	// Copy the seeds into the territories vector
	vector<vector<set<int>>> territoriesLevels;
	vector<set<int>> dummyVector;
	for (int i = 0; i < partition.numSubdomains; i++) {
		dummyVector.push_back(partition.seeds[i]);
		territoriesLevels.push_back(dummyVector);
		dummyVector.clear();
		for (auto seedDOF : partition.seeds[i]) {
			partition.subdomainOfDOFs[seedDOF] = i;
			partition.distanceFromSeed[seedDOF] = 0;
			// printf("Subdomain %d: Adding seed dof %d\n", i, seedDOF); 
		}
	}
	

	// Perform an expansion step for each subdomain, starting from the seed
	int neighbor;
	set<int> setOfNeighborsToAdd;
	for (int iter = 0; iter < numExpansionSteps; iter++) {
		for (int i = 0; i < partition.numSubdomains; i++) {
			uint32_t minLevel = UINT32_MAX;
			for (auto seedDOF : partition.seeds[i]) {
				if (iterationLevel[seedDOF] < minLevel) {
					minLevel = iterationLevel[seedDOF];
					// printf("Subdomain %d: Minlevel = %d\n", i, minLevel);
				}
			}
			for (int level = 0; level < iter+1; level++) {
				for (auto seedDOF : territoriesLevels[i][level]) {
					for (int j = matrix.indexPtr[seedDOF]; j < matrix.indexPtr[seedDOF+1]; j++) {
						neighbor = matrix.nodeNeighbors[j];
						if (partition.subdomainOfDOFs[neighbor] == UINT32_MAX && iterationLevel[neighbor] < iter+1-level) {
							setOfNeighborsToAdd.insert(neighbor);
							partition.subdomainOfDOFs[neighbor] = i;
							partition.distanceFromSeed[neighbor] = level+1; // level+1
							// printf("Iter %d, Level %d: Adding dof %d at distance %d\n", iter, level, neighbor, level+1);
						}
					}
				}
				// if (level == iter) {
				if (level+1 > territoriesLevels[i].size()-1) {
					// territoriesLevels[i].push_back(setOfNeighborsToAdd);
					// changed above to next line
					territoriesLevels[i].push_back(setOfNeighborsToAdd);
                    /* for (auto seed: setOfNeighborsToAdd) {
						printf("Adding dof %d to a new set at the end\n", seed);
					} */
				}
				else {		
					// territoriesLevels[i][level].insert(setOfNeighborsToAdd.begin(), setOfNeighborsToAdd.end());
					// changed above to next line
					territoriesLevels[i][level+1].insert(setOfNeighborsToAdd.begin(), setOfNeighborsToAdd.end());
					/* for (auto seed: setOfNeighborsToAdd) {
						printf("Adding dof %d to the level %d set\n", seed, level);
					} */
				}
				setOfNeighborsToAdd.clear();
			}
		}
	}

	// Concatenate all the contents of each subdomain's territories into a single set
	uint32_t distance;
	set<int> mergedSetInSubdomain;
	for (int i = 0; i < partition.numSubdomains; i++) {
		mergedSetInSubdomain.clear();
		// distance = 0;
		for (auto seedSet : territoriesLevels[i]) {
			mergedSetInSubdomain.insert(seedSet.begin(), seedSet.end());
			/* for (auto seed : seedSet) {
				partition.distanceFromSeedSubdomain[i].push_back(distance);
			}
			distance += 1; */
		}
		partition.territoryDOFsInterior.push_back(mergedSetInSubdomain);
	}
}

void createHaloRegions(meshPartitionForStage &partition, linearSystem matrix)
{
	// Copy territories into territoriesExpanded
	for (int i = 0; i < partition.numSubdomains; i++) {
		partition.territoryDOFsExterior.push_back(partition.territoryDOFsInterior[i]);
	}	

	// For each subdomain, add exterior DOFs to create expanded set, and remove all interior DOFs
	uint32_t neighbor;
	for (int i = 0; i < partition.numSubdomains; i++) {
		// Add all the neigbors of members to create expanded set
		for (auto dof : partition.territoryDOFsInterior[i]) {
			for (int j = matrix.indexPtr[dof]; j < matrix.indexPtr[dof+1]; j++) {
				neighbor = matrix.nodeNeighbors[j];
				partition.territoryDOFsExterior[i].insert(neighbor);
			}
		}
		// Remove all elements which actually belong to the interior
		for (auto dof : partition.territoryDOFsInterior[i]) {
			partition.territoryDOFsExterior[i].erase(dof);
		}
	}
}


// Create territory data and territory Index Ptr data structures and allocate to host/device

void createTerritoriesHost(meshPartitionForStage &partition)
{
	uint32_t numSubdomains = partition.numSubdomains;
	partition.territoryIndexPtr = new uint32_t[numSubdomains+1];
	partition.interiorDOFsPerSubdomain = new uint32_t[numSubdomains];
	partition.territoryIndexPtr[0] = 0;

	for (int i = 0; i < numSubdomains; i++) {
		partition.territoryIndexPtr[i+1] = partition.territoryIndexPtr[i] + (partition.territoryDOFsInterior[i].size() + partition.territoryDOFsExterior[i].size());
		partition.interiorDOFsPerSubdomain[i] = partition.territoryDOFsInterior[i].size();
	}

	uint32_t numElems = partition.territoryIndexPtr[numSubdomains];
	partition.territoryDOFs = new uint32_t[numElems];
	uint32_t idx1 = 0;
	for (int i = 0; i < numSubdomains; i++) {
		for (auto elem : partition.territoryDOFsInterior[i]) {
			partition.territoryDOFs[idx1] = elem;
			// printf("Subdomain %d has element %d\n", i, elem);
			idx1 += 1;
		}
		for (auto elem : partition.territoryDOFsExterior[i]) {
			partition.territoryDOFs[idx1] = elem;
			// printf("Expanded Subdomain %d has element %d\n", i, elem);
			idx1 += 1;
		}
	}
}

void constructTerritoriesHost(meshPartitionForStage &partition, linearSystem matrix, uint32_t * iterationLevel, uint32_t N, uint32_t nSub, uint32_t stageID)
{
	// Determine number of territory expansion steps
	uint32_t numExpansionSteps;
	uint32_t nPerSub = N/nSub;
	if (stageID == 0) numExpansionSteps = (nPerSub-1)/2; // nPerSub-2; // N/3-2; // Originally N/2-2
	if (stageID == 1) numExpansionSteps = (nPerSub-1)/2; // N/3-2; // Originally N/2-2
	if (stageID == 2) numExpansionSteps = ((nPerSub-1)/2)*2+1; // nPerSub-1; // 2*(nPerSub-2); // N-4; // N-6; // originally N-4 // N/2-1;
	if (stageID == 3) numExpansionSteps = ((nPerSub-1)/2)*2+1; // nPerSub-1; // 2*(nPerSub-2); // 2*(nPerSub-2)+1; // N-6; // originally N-3; // N/2; // N/2;

	// Expand the seeds into subdomains
	seedsExpandIntoSubdomains(partition, matrix, iterationLevel, numExpansionSteps);
	createHaloRegions(partition, matrix);

	// Allocate host memory for arrays to be passed to the GPU
	createTerritoriesHost(partition);
}
