// Create territories based on seeds and expand for exterior DOFs

void seedsExpandIntoSubdomains(meshPartitionForStage &partition, linearSystem matrix, uint32_t * iterationLevel, uint32_t numExpansionSteps)
{
	// Initialize subdomain of DOFs and distance from seeds data pointers
	partition.subdomainOfDOFs = new uint32_t[matrix.Ndofs];

	// Load subdomain of DOFs and distance arrays to track who owns which dof and how far they are from seeds (useful info for iterations)
	for (int i = 0; i < matrix.Ndofs; i++) {
		partition.subdomainOfDOFs[i] = UINT32_MAX;
	}

	// Initialize territory DOFs Interior data structures with seeds
	vector<int> seedVector;
	for (int i = 0; i < partition.numSubdomains; i++) {
		for (auto seedDOF : partition.seeds[i]) {
			seedVector.push_back(seedDOF);
			partition.subdomainOfDOFs[seedDOF] = i;
		}
		partition.territoryDOFsInterior.push_back(seedVector);
		seedVector.clear();
	}

	// Expand the subdomains
	vector<int> dofsToExpand;
	for (int iter = 0; iter < numExpansionSteps; iter++) {
		for (int i = 0; i < partition.numSubdomains; i++) {
			// obtain the set in interior and loop through all DOFs
			dofsToExpand = partition.territoryDOFsInterior[i];
			for (auto dof : dofsToExpand) {
				for (int j = matrix.indexPtr[dof]; j < matrix.indexPtr[dof+1]; j++) {
					uint32_t neighbor = matrix.nodeNeighbors[j];
					// if the dof is unclaimed, claim it and place in interior
					if (partition.subdomainOfDOFs[neighbor] == UINT32_MAX && iterationLevel[neighbor] < iter+1) {
						// printf("i = %d, neighbor = %d\n", i, neighbor);
						partition.territoryDOFsInterior[i].push_back(neighbor);
						partition.subdomainOfDOFs[neighbor] = i;
					}
				}
			}
		}
	}
}

/*
void determineIterationLevelPerDOF(meshPartitionForStage &partition, linearSystem matrix, uint32_t * iterationLevel, uint32_t minIters, uint32_t maxIters)
{

	// Initialize the variable (vector of vectors) iterationLevelPerDOF
	for (int i = 0; i < partition.numSubdomains; i++) {
		vector<int> iterationLevelInSubdomain;
		for (auto dof : partition.territoryDOFsInterior[i]) {
			iterationLevelInSubdomain.push_back(iterationLevel[dof]);
		}
		partition.iterationLevelPerDOF.push_back(iterationLevelInSubdomain);
	}

	// Fill in iterationLevelPerDOF with max number of iterations possible per DOF
	for (int i = 0; i < partition.numSubdomains; i++) {
		for (int iter = minIters; iter < maxIters; iter++) {
			// printf("Iteration %d\n", iter);
			vector<int> iterationLevelCopy = partition.iterationLevelPerDOF[i];
			uint32_t idx = 0;
			for (auto dof : partition.territoryDOFsInterior[i]) {
				bool updateDOF = true;
				if (iterationLevelCopy[idx] < iter+1) {
					for (int j = matrix.indexPtr[dof]; j < matrix.indexPtr[dof+1]; j++) {
						uint32_t neighbor = matrix.nodeNeighbors[j];
						// printf("dof %d has neighbor %d\n", dof, neighbor);
						auto iterInterior = find(partition.territoryDOFsInterior[i].begin(), partition.territoryDOFsInterior[i].end(), neighbor);
						auto iterGhost = find(partition.territoryDOFsGhost[i].begin(), partition.territoryDOFsGhost[i].end(), neighbor);
						// Don't update our dof if one of the following is true
						// One of the neighbors is neither in the interior set or the ghost set
						if (iterInterior == partition.territoryDOFsInterior[i].end() && iterGhost == partition.territoryDOFsGhost[i].end()) { 
							updateDOF = false;
						}
						// One of the neighbors is in the interior, but the neighbor level is less than the dof level
						else if (iterInterior != partition.territoryDOFsInterior[i].end()) {
							// printf("We are here\n");					
							if (iterationLevelCopy[iterInterior - partition.territoryDOFsInterior[i].begin()] < iterationLevelCopy[idx]) {
								updateDOF = false;
							}
						}
						// One of the neighbors is in the ghost, but the neighbor level is less than the dof level
						else if (iterGhost != partition.territoryDOFsGhost[i].end()) {
							if (iterationLevel[neighbor] < iterationLevelCopy[idx]) {
								updateDOF = false;
							}
						}
					}
				}
				else {
					updateDOF = false;
				}
				if (updateDOF == true) {
					partition.iterationLevelPerDOF[i][idx] += 1;
					// printf("Updating dof %d in subdomain %d to iter %d\n", dof, i, partition.iterationLevelPerDOF[i][idx]);
				}
				idx += 1;
			}
		}
	}

	// Fill in numDOFsInteriorPerSubdomain and numDOFsToReturnPerSubdomain parameters
	partition.numDOFsInteriorPerSubdomain = new uint32_t[partition.numSubdomains];
	partition.numDOFsToReturnPerSubdomain = new uint32_t[partition.numSubdomains];
	for (int i = 0; i < partition.numSubdomains; i++) {
		partition.numDOFsInteriorPerSubdomain[i] = partition.territoryDOFsInterior[i].size();
		partition.numDOFsToReturnPerSubdomain[i] = partition.territoryDOFsInterior[i].size();
	}

}
*/

/*
void seedsExpandIntoSubdomains(meshPartitionForStage &partition, linearSystem matrix, uint32_t * iterationLevel, uint32_t numExpansionSteps)
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
	set<int> mergedSetInSubdomain;
	for (int i = 0; i < partition.numSubdomains; i++) {
		mergedSetInSubdomain.clear();
		for (auto seedSet : territoriesLevels[i]) {
			mergedSetInSubdomain.insert(seedSet.begin(), seedSet.end());
		}
		partition.territoryDOFsInterior.push_back(mergedSetInSubdomain);
	}
}
*/

/*
void seedsExpandIntoSubdomains3(meshPartitionForStage &partition, linearSystem matrix, uint32_t * iterationLevel, uint32_t numExpansionSteps)
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
		// partition.territoryDOFsInteriorExt.push_back(dummySet);
		// dummySet.clear();
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
							setOfNeighborsToAdd.insert(neighbor);
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
	set<int> mergedSetInSubdomain;
	set<int> interiorDOFs;
	set<int> interiorExtDOFs;
	for (int i = 0; i < partition.numSubdomains; i++) {
		mergedSetInSubdomain.clear();
		interiorDOFs.clear();
		interiorExtDOFs.clear();
		for (auto seedSet : territoriesLevels[i]) {
			mergedSetInSubdomain.insert(seedSet.begin(), seedSet.end());
		}
		for (auto dof : mergedSetInSubdomain) {
			if (partition.subdomainOfDOFs[dof] == i) {
				interiorDOFs.insert(dof);
			}
			else {
				interiorExtDOFs.insert(dof);
			}
		}
		partition.territoryDOFsInterior.push_back(interiorDOFs);
		partition.territoryDOFsInteriorExt.push_back(interiorExtDOFs);
	}
}
*/

/*
void seedsExpandIntoSubdomains2(meshPartitionForStage &partition, linearSystem matrix, uint32_t * iterationLevel, uint32_t numExpansionSteps)
{
	// Initialize subdomain of DOFs and distance from seeds data pointers
	partition.subdomainOfDOFs = new uint32_t[matrix.Ndofs];

	// Load subdomain of DOFs and distance arrays to track who owns which dof and how far they are from seeds (useful info for iterations)
	for (int i = 0; i < matrix.Ndofs; i++) {
		partition.subdomainOfDOFs[i] = UINT32_MAX;
	}

	// Copy the seeds into the territories vector
	vector<vector<set<int>>> territoriesLevels;
	vector<vector<set<int>>> territoriesLevelsInteriorExt;
	vector<set<int>> dummyVector;
	set<int> dummySet;
	for (int i = 0; i < partition.numSubdomains; i++) {
		dummyVector.push_back(partition.seeds[i]);
		territoriesLevels.push_back(dummyVector);
		dummyVector.clear();
		dummyVector.push_back(dummySet);
		territoriesLevelsInteriorExt.push_back(dummyVector);
		dummySet.clear();
		for (auto seedDOF : partition.seeds[i]) {
			partition.subdomainOfDOFs[seedDOF] = i;
			// printf("Subdomain %d: Adding seed dof %d\n", i, seedDOF); 
		}
	}

	// Perform an expansion step for each subdomain, starting from the seed
	int neighbor;
	set<int> setOfNeighborsToAdd;
	set<int> setOfNeighborsToAddInteriorExt;
	for (int iter = 0; iter < numExpansionSteps; iter++) {
		for (int i = 0; i < partition.numSubdomains; i++) {
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
							setOfNeighborsToAddInteriorExt.insert(neighbor);
						}
					}
				}
				for (auto seedDOF : territoriesLevelsInteriorExt[i][level]) {
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
							setOfNeighborsToAddInteriorExt.insert(neighbor);
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
				if (level+1 > territoriesLevelsInteriorExt[i].size()-1) {
					// territoriesLevels[i].push_back(setOfNeighborsToAdd);
					// changed above to next line
					territoriesLevelsInteriorExt[i].push_back(setOfNeighborsToAddInteriorExt);
                    // for (auto seed: setOfNeighborsToAdd) {
					//	printf("Adding dof %d to a new set at the end\n", seed);
					//} 
				}
				else {		
					// territoriesLevels[i][level].insert(setOfNeighborsToAdd.begin(), setOfNeighborsToAdd.end());
					// changed above to next line
					territoriesLevelsInteriorExt[i][level+1].insert(setOfNeighborsToAddInteriorExt.begin(), setOfNeighborsToAddInteriorExt.end());
					// for (auto seed: setOfNeighborsToAdd) {
					//	printf("Adding dof %d to the level %d set\n", seed, level);
					//} 
				}
				setOfNeighborsToAddInteriorExt.clear();
			}
		}
	}

	// Concatenate all the contents of each subdomain's territories into a single set
	set<int> mergedSetInSubdomain;
	for (int i = 0; i < partition.numSubdomains; i++) {
		mergedSetInSubdomain.clear();
		for (auto seedSet : territoriesLevels[i]) {
			mergedSetInSubdomain.insert(seedSet.begin(), seedSet.end());
		}
		partition.territoryDOFsInterior.push_back(mergedSetInSubdomain);
	}
	// Interior Ext
	for (int i = 0; i < partition.numSubdomains; i++) {
		mergedSetInSubdomain.clear();
		for (auto seedSet : territoriesLevelsInteriorExt[i]) {
			mergedSetInSubdomain.insert(seedSet.begin(), seedSet.end());
		}
		partition.territoryDOFsInteriorExt.push_back(mergedSetInSubdomain);
	}
}
*/

/*
void seedsExpandIntoSubdomainsSimple(meshPartitionForStage &partition, linearSystem matrix, uint32_t * iterationLevel, uint32_t numExpansionSteps)
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
*/

/*
void seedsExpandIntoSubdomainsLowerPyramidal(meshPartitionForStage &partition, linearSystem matrix, uint32_t * iterationLevel, uint32_t numExpansionSteps)
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
*/

void createHaloRegions(meshPartitionForStage &partition, linearSystem matrix)
{

	// Copy DOFs into exterior container
	set<int> ghostSet;
	for (int i = 0; i < partition.numSubdomains; i++) {
		ghostSet.clear();
		for (auto elem : partition.territoryDOFsInterior[i]) {
			// printf("%d\n", elem);
			ghostSet.insert(elem);
		}
		partition.territoryDOFsGhost.push_back(ghostSet);
	}
	
	// For each subdomain, add exterior DOFs to create expanded set, and remove all interior DOFs
	uint32_t neighbor;
	vector<set<int>> territoryDOFsGhostCopy = partition.territoryDOFsGhost;
	for (int i = 0; i < partition.numSubdomains; i++) {
		// printf("Subdomain %d\n", i);
		// Add all the neigbors of interiorDOFs to create expanded set
		for (auto dof : partition.territoryDOFsGhost[i]) {
			// printf("dof = %d\n", dof);
			for (int j = matrix.indexPtr[dof]; j < matrix.indexPtr[dof+1]; j++) {
				// printf("j = %d\n", j);
				neighbor = matrix.nodeNeighbors[j];
				territoryDOFsGhostCopy[i].insert(neighbor);
			}
		}
		partition.territoryDOFsGhost[i] = territoryDOFsGhostCopy[i];
		// Remove all elements which actually belong to the interior or interior/ext
		for (auto dof : partition.territoryDOFsInterior[i]) {
			partition.territoryDOFsGhost[i].erase(dof);
		} 
	}

}

// Create territory data and territory Index Ptr data structures and allocate to host/device
void createTerritoriesHost(meshPartitionForStage &partition)
{
	uint32_t numSubdomains = partition.numSubdomains;
	partition.territoryIndexPtr = new uint32_t[numSubdomains+1];
	partition.territoryIndexPtrInterior = new uint32_t[numSubdomains+1];
	partition.territoryIndexPtr[0] = 0;
	partition.territoryIndexPtrInterior[0] = 0;

	for (int i = 0; i < numSubdomains; i++) {
		partition.territoryIndexPtr[i+1] = partition.territoryIndexPtr[i] + (partition.territoryDOFsInterior[i].size() + partition.territoryDOFsGhost[i].size());
		partition.territoryIndexPtrInterior[i+1] = partition.territoryIndexPtrInterior[i] + partition.territoryDOFsInterior[i].size();
	}

	uint32_t numElems = partition.territoryIndexPtr[numSubdomains];
	partition.territoryDOFs = new uint32_t[numElems];
	uint32_t idx1 = 0;
	for (int i = 0; i < numSubdomains; i++) {
		for (auto elem : partition.territoryDOFsInterior[i]) {
			partition.territoryDOFs[idx1] = elem;
			idx1 += 1;
		}
		for (auto elem : partition.territoryDOFsGhost[i]) {
			partition.territoryDOFs[idx1] = elem;
			idx1 += 1;
		}
	}

	uint32_t numElemsInterior = partition.territoryIndexPtrInterior[numSubdomains];
	partition.iterationLevelDOFs = new uint32_t[numElemsInterior];
	uint32_t idx2 = 0;
	for (int i = 0; i < numSubdomains; i++) {
		for (auto elem : partition.iterationLevelPerDOF[i]) {
			partition.iterationLevelDOFs[idx2] = elem;
			idx2 += 1;
		}
	}
}

/*
void determineInvertedIterationsUpdated(meshPartitionForStage &partition, meshPartitionForStage &bridge, uint32_t * iterationLevelUpperPyramidal, uint32_t * iterationLevelBridge, uint32_t maxIters)
{
	// Loop over all DOFs to be updated in each subdomain, and transfer maximum Iters to partition data structure
	uint32_t entryIdx = 0;
	uint32_t numEntries = partition.territoryIndexPtrInteriorExt[partition.numSubdomains];
	partition.maximumIterations = new uint32_t[numEntries];
	for (int i = 0; i < partition.numSubdomains; i++) {
		uint32_t numDOFsToUpdate = partition.territoryIndexPtrInteriorExt[i+1] - partition.territoryIndexPtrInteriorExt[i];	
		uint32_t shift = partition.territoryIndexPtr[i];	
		for (int idx = 0; idx < numDOFsToUpdate; idx++) {
			uint32_t dof = partition.territoryDOFs[shift + idx];
			uint32_t iterCurr = maxIters + (maxIters - iterationLevelBridge[dof]);
			uint32_t deltaIter = bridge.maximumIterations[entryIdx] - iterationLevelUpperPyramidal[dof]; 
			partition.maximumIterations[entryIdx] = iterCurr + deltaIter;
			printf("Subdomain %d: DOF = %d w/ delta_iter = %d, MaxIter = %d\n", i, dof, deltaIter, partition.maximumIterations[entryIdx]);
			entryIdx++;
		}
	}	
}
*/

/*
void determineInvertedIterations(meshPartitionForStage &partition, linearSystem matrix, uint32_t * iterationLevel, uint32_t maxIters)
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
*/

/*
void determineMaximumIterationsPerDOF(meshPartitionForStage &partition, linearSystem matrix, uint32_t * iterationLevel, uint32_t maxIters)
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

void readMaximumIterationsFromFile(meshPartitionForStage &partition, uint32_t Ndofs)
{
	// Read iteration levels from file
	vector<int> iterationLevelFromFile;
	std::ifstream iterationLevelFile("Unstructured_Mesh/Square_Mesh/Seeds/iterationLevel_1.txt");
	uint32_t level;
	for (int i = 0; i < Ndofs; i++) {
		iterationLevelFile >> level;
		iterationLevelFromFile.push_back(level);
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
			partition.maximumIterations[entryIdx] = iterationLevelFromFile[dof];
			// printf("Subdomain %d: DOF = %d @ Level = %d, MaxIter = %d\n", i, dof, maximumIterationsMap[i][dof], partition.maximumIterations[entryIdx]);
			entryIdx++;
		}
	}	
}
*/

/*
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
*/
