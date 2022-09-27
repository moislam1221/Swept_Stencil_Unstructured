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
