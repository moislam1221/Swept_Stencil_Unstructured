// Create territories based on seeds and expand for exterior DOFs

void determineInvertedIterations(meshPartitionForStageNew &partition, linearSystem matrix, uint32_t * iterationLevel, uint32_t maxIters)
{
	// Initialize the maximumIterationMap
	vector<map<int, int>> maximumIterationsMap;
	map<int, int> maximumItersInSubdomain;
	for (int i = 0; i < partition.numSubdomains; i++) {
		maximumItersInSubdomain.clear();
		for (int idx = partition.territoryIndexPtr[i]; idx < partition.territoryIndexPtr[i+1]; idx++) {
			uint32_t dof = partition.territoryDOFs[idx];
			maximumItersInSubdomain[dof] = maxIters + (maxIters - iterationLevel[dof]);
		}
		maximumIterationsMap.push_back(maximumItersInSubdomain);
	}	

	// Loop over all DOFs to be updated in each subdomain, and transfer maximum Iters to partition data structure
	uint32_t entryIdx = 0;
	uint32_t numEntries = partition.territoryIndexPtrInteriorExt[partition.numSubdomains];
	partition.maximumIterations = new uint32_t[numEntries];
	for (int i = 0; i < partition.numSubdomains; i++) {
		uint32_t numDOFsToUpdate = partition.territoryIndexPtrInteriorExt[i+1] - partition.territoryIndexPtrInteriorExt[i];	
		uint32_t shift = partition.territoryIndexPtr[i];	
		for (int idx = 0; idx < numDOFsToUpdate; idx++) {
			uint32_t dof = partition.territoryDOFs[shift + idx];
			partition.maximumIterations[entryIdx] = maximumIterationsMap[i][dof];
			// printf("Subdomain %d: DOF = %d @ Level = %d, MaxIter = %d\n", i, dof, maximumIterationsMap[i][dof], partition.maximumIterations[entryIdx]);
			entryIdx++;
		}
	}	
}

void determineMaximumIterationsPerDOF(meshPartitionForStageNew &partition, linearSystem matrix, uint32_t * iterationLevel, uint32_t maxIters)
{
	// Initialize the maximumIterationMap
	vector<map<int, int>> maximumIterationsMap;
	map<int, int> maximumItersInSubdomain;
	for (int i = 0; i < partition.numSubdomains; i++) {
		maximumItersInSubdomain.clear();
		for (int idx = partition.territoryIndexPtr[i]; idx < partition.territoryIndexPtr[i+1]; idx++) {
			uint32_t dof = partition.territoryDOFs[idx];
			maximumItersInSubdomain[dof] = iterationLevel[dof];
		}
		maximumIterationsMap.push_back(maximumItersInSubdomain);
	}	

	// Determine the maximum number of iterations I can perform on each DOF on each subdomain
	map<int, int>::iterator neighborIterator;
	for (int i = 0; i < partition.numSubdomains; i++) {
		map<int, int>subdomainMapCopy = maximumIterationsMap[i];
		// bool continueUpdate = true;
		// while (continueUpdate) { 
		for (int iter = 0; iter < maxIters; iter++) {
			// continueUpdate = false;
			// for (auto entry = maximumIterationsMap[i].begin(); entry != maximumIterationsMap[i].end(); entry++) {
			uint32_t numDOFsToUpdate = partition.territoryIndexPtrInteriorExt[i+1] - partition.territoryIndexPtrInteriorExt[i];
			uint32_t shift = partition.territoryIndexPtr[i]; 
			for (int idx = 0; idx < numDOFsToUpdate; idx++) {
				uint32_t dof = partition.territoryDOFs[shift + idx];
				bool updateDOF = true;
				if (maximumIterationsMap[i][dof] <= iter) {
					for (int idx = matrix.indexPtr[dof]; idx < matrix.indexPtr[dof+1]; idx++) {
						uint32_t neighbor = matrix.nodeNeighbors[idx];
						neighborIterator = maximumIterationsMap[i].find(neighbor);
						if (neighborIterator == maximumIterationsMap[i].end()) {
							updateDOF = false;
						}
						else if (maximumIterationsMap[i][neighbor] < maximumIterationsMap[i][dof]) { 
							updateDOF = false;
						}
					}
				}
				else {
					updateDOF = false;
				}
				if (updateDOF == true) {
					subdomainMapCopy[dof] = subdomainMapCopy[dof] + 1;
					// continueUpdate = true;
				}
			}
			maximumIterationsMap[i] = subdomainMapCopy;
		}
	}

	// Loop over all DOFs to be updated in each subdomain, and transfer maximum Iters to partition data structure
	uint32_t entryIdx = 0;
	uint32_t numEntries = partition.territoryIndexPtrInteriorExt[partition.numSubdomains];
	partition.maximumIterations = new uint32_t[numEntries];
	for (int i = 0; i < partition.numSubdomains; i++) {
		uint32_t numDOFsToUpdate = partition.territoryIndexPtrInteriorExt[i+1] - partition.territoryIndexPtrInteriorExt[i];	
		uint32_t shift = partition.territoryIndexPtr[i];	
		for (int idx = 0; idx < numDOFsToUpdate; idx++) {
			uint32_t dof = partition.territoryDOFs[shift + idx];
			partition.maximumIterations[entryIdx] = maximumIterationsMap[i][dof];
			// printf("Subdomain %d: DOF = %d @ Level = %d, MaxIter = %d\n", i, dof, maximumIterationsMap[i][dof], partition.maximumIterations[entryIdx]);
			entryIdx++;
		}
	}	

}

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
			/* for (auto seedDOF : partition.seeds[i]) {
				if (iterationLevel[seedDOF] < minLevel) {
					minLevel = iterationLevel[seedDOF];
					// printf("Subdomain %d: Minlevel = %d\n", i, minLevel);
				}
			} */
			for (int level = 0; level < iter+1; level++) {
				for (auto seedDOF : territoriesLevels[i][level]) {
					// printf("seedDOF is %d\n", seedDOF);
					for (int j = matrix.indexPtr[seedDOF]; j < matrix.indexPtr[seedDOF+1]; j++) {
						neighbor = matrix.nodeNeighbors[j];
						// printf("neighbor is %d\n", neighbor);
						if (partition.subdomainOfDOFs[neighbor] == UINT32_MAX && iterationLevel[neighbor] < iter+1-level) {
							setOfNeighborsToAdd.insert(neighbor);
							partition.subdomainOfDOFs[neighbor] = i;
							partition.distanceFromSeed[neighbor] = level+1; // level+1
							// printf("Subdomain %d: Iter %d, Level %d: Adding dof %d at distance %d\n", i, iter, level, neighbor, level+1);
						}
					}
				}
				// if (level == iter) {
				if (level+1 > territoriesLevels[i].size()-1) {
					// territoriesLevels[i].push_back(setOfNeighborsToAdd);
					// changed above to next line
					territoriesLevels[i].push_back(setOfNeighborsToAdd);
                    // for (auto seed: setOfNeighborsToAdd) {
					//	printf("Adding dof %d to a new set at the end\n", seed);
					//} 
				}
				else {		
					// territoriesLevels[i][level].insert(setOfNeighborsToAdd.begin(), setOfNeighborsToAdd.end());
					// changed above to next line
					territoriesLevels[i][level+1].insert(setOfNeighborsToAdd.begin(), setOfNeighborsToAdd.end());
					// for (auto seed: setOfNeighborsToAdd) {
					//	printf("Adding dof %d to the level %d set\n", seed, level);
					//} 
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
			// for (auto seed : seedSet) {
			//	partition.distanceFromSeedSubdomain[i].push_back(distance);
			//}
			// distance += 1; 
		}
		partition.territoryDOFsInterior.push_back(mergedSetInSubdomain);
	}

}

void seedsExpandIntoSubdomainsNew(meshPartitionForStageNew &partition, linearSystem matrix, uint32_t * iterationLevel, uint32_t numExpansionSteps)
{
	// Initialize subdomain of DOFs and distance from seeds data pointers
	partition.subdomainOfDOFs = new uint32_t[matrix.Ndofs];

	// Load subdomain of DOFs and distance arrays to track who owns which dof and how far they are from seeds (useful info for iterations)
	for (int i = 0; i < matrix.Ndofs; i++) {
		partition.subdomainOfDOFs[i] = UINT32_MAX;
	}

	// Copy the seeds into the territories vector
	vector<vector<set<int>>> territoriesLevels;
	vector<set<int>> dummyVector;
	set<int> dummySet;
	for (int i = 0; i < partition.numSubdomains; i++) {
		dummyVector.push_back(partition.seeds[i]);
		territoriesLevels.push_back(dummyVector);
		dummyVector.clear();
		partition.territoryDOFsInteriorExt.push_back(dummySet);
		dummySet.clear();
		for (auto seedDOF : partition.seeds[i]) {
			partition.subdomainOfDOFs[seedDOF] = i;
			// printf("Subdomain %d: Adding seed dof %d\n", i, seedDOF); 
		}
	}

	// Perform an expansion step for each subdomain, starting from the seed
	int neighbor;
	set<int> setOfNeighborsToAdd;
	for (int iter = 0; iter < numExpansionSteps; iter++) {
		for (int i = 0; i < partition.numSubdomains; i++) {
			uint32_t minLevel = UINT32_MAX;
			/* for (auto seedDOF : partition.seeds[i]) {
				if (iterationLevel[seedDOF] < minLevel) {
					minLevel = iterationLevel[seedDOF];
					// printf("Subdomain %d: Minlevel = %d\n", i, minLevel);
				}
			} */
			for (int level = 0; level < iter+1; level++) {
				for (auto seedDOF : territoriesLevels[i][level]) {
					// printf("seedDOF is %d\n", seedDOF);
					for (int j = matrix.indexPtr[seedDOF]; j < matrix.indexPtr[seedDOF+1]; j++) {
						neighbor = matrix.nodeNeighbors[j];
						// printf("neighbor is %d\n", neighbor);
						if (partition.subdomainOfDOFs[neighbor] == UINT32_MAX && iterationLevel[neighbor] < iter+1-level) {
							setOfNeighborsToAdd.insert(neighbor);
							partition.subdomainOfDOFs[neighbor] = i;
							// printf("Subdomain %d: Iter %d, Level %d: Adding dof %d\n", i, iter, level, neighbor);
						}
						else if (partition.subdomainOfDOFs[neighbor] != i && iterationLevel[neighbor] < iter+1-level) {
							partition.territoryDOFsInteriorExt[i].insert(neighbor);	
						}
					}
				}
				// if (level == iter) {
				if (level+1 > territoriesLevels[i].size()-1) {
					// territoriesLevels[i].push_back(setOfNeighborsToAdd);
					// changed above to next line
					territoriesLevels[i].push_back(setOfNeighborsToAdd);
                    // for (auto seed: setOfNeighborsToAdd) {
					//	printf("Adding dof %d to a new set at the end\n", seed);
					//} 
				}
				else {		
					// territoriesLevels[i][level].insert(setOfNeighborsToAdd.begin(), setOfNeighborsToAdd.end());
					// changed above to next line
					territoriesLevels[i][level+1].insert(setOfNeighborsToAdd.begin(), setOfNeighborsToAdd.end());
					// for (auto seed: setOfNeighborsToAdd) {
					//	printf("Adding dof %d to the level %d set\n", seed, level);
					//} 
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
			// for (auto seed : seedSet) {
			//	partition.distanceFromSeedSubdomain[i].push_back(distance);
			//}
			// distance += 1; 
		}
		partition.territoryDOFsInterior.push_back(mergedSetInSubdomain);
	}

}


void seedsExpandIntoSubdomainsNewSimple(meshPartitionForStageNew &partition, linearSystem matrix, uint32_t * iterationLevel, uint32_t numExpansionSteps)
{
	// Initialize subdomain of DOFs and distance from seeds data pointers
	partition.subdomainOfDOFs = new uint32_t[matrix.Ndofs];

	// Load subdomain of DOFs and distance arrays to track who owns which dof and how far they are from seeds (useful info for iterations)
	for (int i = 0; i < matrix.Ndofs; i++) {
		partition.subdomainOfDOFs[i] = UINT32_MAX;
	}

	// Initialize territory DOFs Interior and territory DOFs Interior/Ext data structures
	set<int> emptySet;
	for (int i = 0; i < partition.numSubdomains; i++) {
		partition.territoryDOFsInterior.push_back(partition.seeds[i]);
		partition.territoryDOFsInteriorExt.push_back(emptySet);
		for (auto seedDOF : partition.seeds[i]) {
			partition.subdomainOfDOFs[seedDOF] = i;
		}
	}

	// Expand the subdomains
	set<int> dofsToExpand;
	for (int iter = 0; iter < numExpansionSteps; iter++) {
		for (int i = 0; i < partition.numSubdomains; i++) {
			// combine the sets in interior + interio/ext and loop through all DOFs
			dofsToExpand.clear();
			dofsToExpand.insert(partition.territoryDOFsInterior[i].begin(), partition.territoryDOFsInterior[i].end());
			dofsToExpand.insert(partition.territoryDOFsInteriorExt[i].begin(), partition.territoryDOFsInteriorExt[i].end());
			for (auto dof : dofsToExpand) {
				for (int j = matrix.indexPtr[dof]; j < matrix.indexPtr[dof+1]; j++) {
					uint32_t neighbor = matrix.nodeNeighbors[j];
					// if the dof is unclaimed, claim it and place in interior
					if (partition.subdomainOfDOFs[neighbor] == UINT32_MAX) {
						partition.territoryDOFsInterior[i].insert(neighbor);
						partition.subdomainOfDOFs[neighbor] = i;
					}
					// if the dof is claimed, place it in interior/ext
					else if (partition.subdomainOfDOFs[neighbor] != i) {
						partition.territoryDOFsInteriorExt[i].insert(neighbor);	
					}
				}
			}
		}
	}
}


void seedsExpandIntoSubdomainsNewLowerPyramidal(meshPartitionForStageNew &partition, linearSystem matrix, uint32_t * iterationLevel, uint32_t numExpansionSteps)
{
	// Initialize subdomain of DOFs and distance from seeds data pointers
	partition.subdomainOfDOFs = new uint32_t[matrix.Ndofs];

	// Load subdomain of DOFs and distance arrays to track who owns which dof and how far they are from seeds (useful info for iterations)
	for (int i = 0; i < matrix.Ndofs; i++) {
		partition.subdomainOfDOFs[i] = UINT32_MAX;
	}

	// Initialize territory DOFs Interior and territory DOFs Interior/Ext data structures
	set<int> emptySet;
	for (int i = 0; i < partition.numSubdomains; i++) {
		partition.territoryDOFsInterior.push_back(partition.seeds[i]);
		partition.territoryDOFsInteriorExt.push_back(emptySet);
		for (auto seedDOF : partition.seeds[i]) {
			partition.subdomainOfDOFs[seedDOF] = i;
		}
	}

	// Expand the subdomains
	set<int> territoryDOFsInteriorCopy;
	for (int level = 1; level < numExpansionSteps+1; level++) {
		for (int i = 0; i < partition.numSubdomains; i++) {
			// combine the sets in interior + interio/ext and loop through all DOFs
			territoryDOFsInteriorCopy.insert(partition.territoryDOFsInterior[i].begin(), partition.territoryDOFsInterior[i].end());
			while (territoryDOFsInteriorCopy.size() > 0) {
				uint32_t dof = *territoryDOFsInteriorCopy.begin();
				territoryDOFsInteriorCopy.erase(dof);
				for (int j = matrix.indexPtr[dof]; j < matrix.indexPtr[dof+1]; j++) {
					uint32_t neighbor = matrix.nodeNeighbors[j];
					// if the dof is unclaimed, claim it and place in interior
					if (partition.subdomainOfDOFs[neighbor] == UINT32_MAX && iterationLevel[neighbor] < level) {
						partition.territoryDOFsInterior[i].insert(neighbor);
						territoryDOFsInteriorCopy.insert(neighbor);
						partition.subdomainOfDOFs[neighbor] = i;
					}
					// if the dof is claimed, place it in interior/ext
					// else if (partition.subdomainOfDOFs[neighbor] != i) {
					//	partition.territoryDOFsInteriorExt[i].insert(neighbor);	
					//}
				}
			}
		}
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

void createHaloRegionsNew(meshPartitionForStageNew &partition, linearSystem matrix)
{
	// Copy interior or interior/ext DOFs into exterior container
	set<int> setOfInteriorDOFs;
	for (int i = 0; i < partition.numSubdomains; i++) {
		setOfInteriorDOFs.clear();
		setOfInteriorDOFs.insert(partition.territoryDOFsInterior[i].begin(), partition.territoryDOFsInterior[i].end());
		setOfInteriorDOFs.insert(partition.territoryDOFsInteriorExt[i].begin(), partition.territoryDOFsInteriorExt[i].end());
		partition.territoryDOFsExterior.push_back(setOfInteriorDOFs);
	}	

	// For each subdomain, add exterior DOFs to create expanded set, and remove all interior DOFs
	uint32_t neighbor;
	vector<set<int>> territoryDOFsExteriorCopy = partition.territoryDOFsExterior;
	for (int i = 0; i < partition.numSubdomains; i++) {
		// Add all the neigbors of interiorDOFs to create expanded set
		for (auto dof : partition.territoryDOFsExterior[i]) {
			for (int j = matrix.indexPtr[dof]; j < matrix.indexPtr[dof+1]; j++) {
				neighbor = matrix.nodeNeighbors[j];
				territoryDOFsExteriorCopy[i].insert(neighbor);
			}
		}
		partition.territoryDOFsExterior[i] = territoryDOFsExteriorCopy[i];
		// Remove all elements which actually belong to the interior or interior/ext
		for (auto dof : partition.territoryDOFsInterior[i]) {
			partition.territoryDOFsExterior[i].erase(dof);
		}
		for (auto dof : partition.territoryDOFsInteriorExt[i]) {
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
			printf("Subdomain %d: Interior DOF %d\n", i, elem);
			idx1 += 1;
		}
		for (auto elem : partition.territoryDOFsExterior[i]) {
			partition.territoryDOFs[idx1] = elem;
			printf("Subdomain %d: Exterior DOF %d\n", i, elem);
			// printf("Expanded Subdomain %d has element %d\n", i, elem);
			idx1 += 1;
		}
	}
}

void createTerritoriesHostNew(meshPartitionForStageNew &partition)
{
	uint32_t numSubdomains = partition.numSubdomains;
	partition.territoryIndexPtr = new uint32_t[numSubdomains+1];
	partition.territoryIndexPtrInteriorExt = new uint32_t[numSubdomains+1];
	partition.territoryIndexPtrInterior = new uint32_t[numSubdomains+1];
	partition.territoryIndexPtr[0] = 0;
	partition.territoryIndexPtrInteriorExt[0] = 0;
	partition.territoryIndexPtrInterior[0] = 0;

	for (int i = 0; i < numSubdomains; i++) {
		partition.territoryIndexPtr[i+1] = partition.territoryIndexPtr[i] + (partition.territoryDOFsInterior[i].size() + partition.territoryDOFsExterior[i].size()) + partition.territoryDOFsInteriorExt[i].size();
		partition.territoryIndexPtrInteriorExt[i+1] = partition.territoryIndexPtrInteriorExt[i] + (partition.territoryDOFsInterior[i].size() + partition.territoryDOFsInteriorExt[i].size());
		partition.territoryIndexPtrInterior[i+1] = partition.territoryIndexPtrInterior[i] + partition.territoryDOFsInterior[i].size();
	}

	uint32_t numElems = partition.territoryIndexPtr[numSubdomains];
	partition.territoryDOFs = new uint32_t[numElems];
	uint32_t idx1 = 0;
	uint32_t idx2 = 0;
	for (int i = 0; i < numSubdomains; i++) {
		for (auto elem : partition.territoryDOFsInterior[i]) {
			partition.territoryDOFs[idx1] = elem;
			// find the position of elem in the map, and then the ground, and put ground in array
			// printf("Subdomain %d: Interior DOF %d\n", i, elem);
			idx1 += 1;
			idx2 += 1;
		}
		for (auto elem : partition.territoryDOFsInteriorExt[i]) {
			partition.territoryDOFs[idx1] = elem;
			// find the position of elem in the map, and then the ground, and put ground in array
			// printf("Subdomain %d: Interior/Exterior DOF %d\n", i, elem);
			idx1 += 1;
			idx2 += 1;
		}
		for (auto elem : partition.territoryDOFsExterior[i]) {
			partition.territoryDOFs[idx1] = elem;
			// printf("Subdomain %d: Exterior DOF %d\n", i, elem);
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
	if (stageID == 0) numExpansionSteps = (nPerSub-1)/2-1; // nPerSub-2; // N/3-2; // Originally N/2-2
	if (stageID == 1) numExpansionSteps = (nPerSub-1)/2-1; // N/3-2; // Originally N/2-2
	if (stageID == 2) numExpansionSteps = ((nPerSub-1)/2)*2-1; // nPerSub-1; // 2*(nPerSub-2); // N-4; // N-6; // originally N-4 // N/2-1;
	if (stageID == 3) numExpansionSteps = ((nPerSub-1)/2)*2-1; // nPerSub-1; // 2*(nPerSub-2); // 2*(nPerSub-2)+1; // N-6; // originally N-3; // N/2; // N/2;

	// Expand the seeds into subdomains
	seedsExpandIntoSubdomains(partition, matrix, iterationLevel, numExpansionSteps);
	createHaloRegions(partition, matrix);

	// Allocate host memory for arrays to be passed to the GPU
	createTerritoriesHost(partition);
}
