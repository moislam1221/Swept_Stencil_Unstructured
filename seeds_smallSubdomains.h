void fillInSeeds1D(vector<int> &seeds1D, uint32_t N, uint32_t nSub1D) 
{
	// Insert seed 0
	seeds1D.push_back(0);
	// Insert seeds 1...N-2	
	uint32_t nPerSub = ceil(N / nSub1D);
	// printf("nPerSub = %d\n", nPerSub);
	// printf("nSub1D = %d\n", nSub1D);
	uint32_t seedID;
	for (int i = 1; i < nSub1D-1; i++) {
		seedID = i * nPerSub + (nPerSub - 1) / 2;
		seeds1D.push_back(seedID);
	}
	
	// Insert seed N
	seeds1D.push_back(N-1);
}

void constructSeeds2DFrom1D(vector<int> &seeds2D, vector<int> seeds1D, uint32_t N)
{
	uint32_t nSeeds1D = seeds1D.size();
	uint32_t seedX, seedY, seedID;
	for (int j = 0; j < nSeeds1D; j++) {
		for (int i = 0; i < nSeeds1D; i++) {
			seedX = seeds1D[i];
			seedY = seeds1D[j];
			seedID = seedX + N * seedY;
			// printf("i = %d, j = %d, seedID = %d\n", i, j, seedID);
			seeds2D.push_back(seedID);
		}
	}
}

void constructSeedsUpperPyramidal(meshPartitionForStage &partition, uint32_t N, uint32_t nSub1D)
{
	// nPerSub
	uint32_t nPerSub = N / nSub1D;

	// Initialize vector containing indices in 1D direction of seeds
	vector<int> seeds1D;
	fillInSeeds1D(seeds1D, N, nSub1D);
	
	// Fill in partition.seeds with all seed IDs (1D => 2D but with a flattened 1D index)
	// Initialize seed set variable
	set<int> seedSet;
	uint32_t seedX, seedY, seedID;
	for (int j = 0; j < nSub1D; j++) {
		for (int i = 0; i < nSub1D; i++) {
			seedX = seeds1D[i];
			seedY = seeds1D[j];
			seedID = seedX + N * seedY;
			seedSet.insert(seedID);
			// printf("i = %d, j = %d, seedID = %d\n", i, j, seedID);
			if (i == 0 && j != 0 && j!= nSub1D-1 && nPerSub % 2 == 0) {
				seedSet.insert(seedID+N);
				// printf("seedX = %d, seedY = %d, seedID = %d\n", seedX, seedY, seedID+1);
			}
			else if (i == nSub1D-1 && j != 0 && j!= nSub1D-1 && nPerSub % 2 == 0) {
				seedSet.insert(seedID+N);
				// printf("seedX = %d, seedY = %d, seedID = %d\n", seedX, seedY, seedID-1);
			}
			else if (j == 0 && i != 0 && i != nSub1D-1 && nPerSub % 2 == 0) {
				seedSet.insert(seedID+1);
				// printf("seedX = %d, seedY = %d, seedID = %d\n", seedX, seedY, seedID+N);
			}
			else if (j == nSub1D-1 && i != 0 && i != nSub1D-1 && nPerSub % 2 == 0) {
				seedSet.insert(seedID+1);
				// printf("seedX = %d, seedY = %d, seedID = %d\n", seedX, seedY, seedID-N);
			} 
			else if (i != 0 && j!= 0 && i != nSub1D-1 && j != nSub1D-1 && nPerSub % 2 == 0) {
				seedSet.insert(seedID+1);
				seedSet.insert(seedID+N);
				seedSet.insert(seedID+N+1);
				// printf("seedX = %d, seedY = %d, seedID = %d\n", seedX, seedY, seedID-N);
			}
			partition.seeds.push_back(seedSet);
			seedSet.clear();
		}
	}
}

void constructSeedsBridge(meshPartitionForStage &partition, uint32_t N, uint32_t nSub1D)
{
	// Initialize vector containing indices in 1D direction of seeds
	vector<int> seeds1D;
	fillInSeeds1D(seeds1D, N, nSub1D);
	
	// Fill in partition.seeds with all seed IDs (1D => 2D but with a flattened 1D index)
	vector<int> seeds2D;
	constructSeeds2DFrom1D(seeds2D, seeds1D, N);

	// Initialize seed set
	set<int> seedSet;
	for (int i = 0; i < nSub1D * nSub1D; i++) {
		uint32_t seedBegin = seeds2D[i]; 


		// Horizontal connector seeds
		if (i % nSub1D != nSub1D-1) {
			uint32_t seedEnd = seeds2D[i+1];
			// printf("(%d, %d)\n", seedBegin, seedEnd);
			for (int j = seedBegin+1; j < seedEnd; j++) {
				// printf("%d\n", j);
				seedSet.insert(j);
			}
			partition.seeds.push_back(seedSet);
			seedSet.clear();
		}
		// Vertical connector seeds
		if (i < nSub1D * (nSub1D-1)) {
			uint32_t seedEnd = seeds2D[i+nSub1D]; 
			// printf("(%d, %d)\n", seedBegin, seedEnd);
			for (int j = seedBegin+N; j < seedEnd; j += N) {
				// printf("%d\n", j);
				seedSet.insert(j);
			}
			partition.seeds.push_back(seedSet);
			seedSet.clear();
		}
	}
}

void constructSeedsLowerPyramidal(meshPartitionForStage &partition, uint32_t N, uint32_t nSub1D)
{
	// nPerSub
	uint32_t nPerSub = N / nSub1D;
	
	// Initialize vector containing indices in 1D direction of seeds
	vector<int> seeds1D;
	fillInSeeds1D(seeds1D, N, nSub1D);
	
	// Fill in partition.seeds with all seed IDs (1D => 2D but with a flattened 1D index)
	// Initialize seed set variable
	vector<int> seeds1DMid;
	uint32_t seedID;
	for (int i = 0; i < nSub1D-1; i++) {
		seeds1DMid.push_back((seeds1D[i] + seeds1D[i+1]) / 2);
		// printf("i = %d, seedID = %d\n", i, seeds1DMid[i]);
	}

	// Fill in partition.seeds with all seed IDs (1D => 2D but with a flattened 1D index)
	// Initialize seed set variable
	set<int> seedSet;
	uint32_t seedX, seedY;
	for (int j = 0; j < nSub1D-1; j++) {
		for (int i = 0; i < nSub1D-1; i++) {
			seedX = seeds1DMid[i];
			seedY = seeds1DMid[j];
			seedID = seedX + N * seedY;
			seedSet.insert(seedID);
			// printf("i = %d, j = %d, seedID = %d\n", i, j, seedID);
			// For 2 by 2 case, this seed seems to be 2x2. For 3 by 3 case, this seed seems to be 1x1 so no need to enter here
			if (i == 0 && j != 0 && j!= (nSub1D-1)-1 && nSub1D % 2 == 0) {
				seedSet.insert(seedID+N);
				// printf("seedX = %d, seedY = %d, seedID = %d\n", seedX, seedY, seedID+1);
			}
			else if (i == (nSub1D-1)-1 && j != 0 && j!= (nSub1D-1)-1 && nSub1D % 2 == 0) {
				seedSet.insert(seedID+N);
				// printf("seedX = %d, seedY = %d, seedID = %d\n", seedX, seedY, seedID-1);
			}
			else if (j == 0 && i != 0 && i != (nSub1D-1)-1 && nSub1D % 2 == 0) {
				seedSet.insert(seedID+1);
				// printf("seedX = %d, seedY = %d, seedID = %d\n", seedX, seedY, seedID+N);
			}
			else if (j == (nSub1D-1)-1 && i != 0 && i != (nSub1D-1)-1 && nSub1D % 2 == 0) {
				seedSet.insert(seedID+1);
				// printf("seedX = %d, seedY = %d, seedID = %d\n", seedX, seedY, seedID-N);
			} 
			else if (i != 0 && j!= 0 && i != (nSub1D-1)-1 && j != (nSub1D-1)-1 && nSub1D % 2 == 0) {
				seedSet.insert(seedID+1);
				seedSet.insert(seedID+N);
				seedSet.insert(seedID+N+1);
				// printf("seedX = %d, seedY = %d, seedID = %d\n", seedX, seedY, seedID-N);
			}
			else if (nSub1D == 2) {
				seedSet.insert(seedID+1);
				seedSet.insert(seedID+N);
				seedSet.insert(seedID+N+1);
				// printf("seedX = %d, seedY = %d, seedID = %d\n", seedX, seedY, seedID-N);
			}




			/* if (nPerSub % 2 == 0) {
				printf("Here\n");
				seedSet.insert(seedID+1);
				seedSet.insert(seedID+N);
				seedSet.insert(seedID+N+1);
			} */
			// Added this for N = 15, nSub = 5
			/*if (nSub1D % 2 == 1) {
				seedSet.insert(seedID+1);
				seedSet.insert(seedID+N);
				seedSet.insert(seedID+N+1);
			}*/
			partition.seeds.push_back(seedSet);
			seedSet.clear();
		}
	}
}

void constructSeedsDualBridge(meshPartitionForStage &partition, uint32_t N, uint32_t nSub1D)
{
	// Initialize vector containing indices in 1D direction of seeds
	vector<int> seeds1D;
	fillInSeeds1D(seeds1D, N, nSub1D);
	
	// Fill in partition.seeds with all seed IDs (1D => 2D but with a flattened 1D index)
	// Initialize seed set variable
	vector<int> seeds1DMid;
	uint32_t seedID;
	for (int i = 0; i < nSub1D-1; i++) {
		seeds1DMid.push_back((seeds1D[i] + seeds1D[i+1]) / 2);
		// printf("i = %d, seedID = %d\n", i, seeds1DMid[i]);
	}
	
	// Fill in partition.seeds with all seed IDs (1D => 2D but with a flattened 1D index)
	vector<int> seeds2D;
	constructSeeds2DFrom1D(seeds2D, seeds1DMid, N);

	// Construct dual bridges
	set<int> seedSet;
	uint32_t nSeedsMid1D = seeds1DMid.size();
	uint32_t seedBeginID, seedEndID, seedBegin, seedEnd;
	for (int j = 0; j < nSeedsMid1D; j++) {
		for (int i = 0; i < nSeedsMid1D; i++) {
			seedBeginID = i + j * nSeedsMid1D;
			seedBegin = seeds2D[seedBeginID];
			// Create horizontal bridges between seeds
			if (i != nSeedsMid1D - 1) {
				seedEndID = (i+1) + j * nSeedsMid1D;
				seedEnd = seeds2D[seedEndID];
				for (int k = seedBegin+1; k < seedEnd; k++) {
					seedSet.insert(k);
					if (nSub1D % 2 == 0) {
						seedSet.insert(k+1);
					}
					// printf("1 - Inserting %d\n", k);
				}
				// printf("DONE\n");
				partition.seeds.push_back(seedSet);
				seedSet.clear();
			}
			// Create vertical bridges between seeds
			if (j != nSeedsMid1D - 1) {
				seedEndID = i + (j+1) * nSeedsMid1D;
				seedEnd = seeds2D[seedEndID];
				for (int k = seedBegin+N; k < seedEnd; k+=N) {
					seedSet.insert(k);
					if (nSub1D % 2 == 0) {
						seedSet.insert(k+N);
					}
					// printf("2 - Inserting %d\n", k);
				}
				// printf("DONE\n");
				partition.seeds.push_back(seedSet);
				seedSet.clear();
			}
		}
	}

	uint32_t seed, seed2D, x, y;	
	for (int j = 0; j < nSeedsMid1D; j++) {
		for (int i = 0; i < nSeedsMid1D; i++) {
			// Determine the seed location and x and y coordinates
			seedID = i + j * nSeedsMid1D;
			seed = seeds2D[seedID];
			x = seed % N;
			y = seed / N;
			// Create upper bridges on top seed DOFs
			if (j == 0) {
				for (int k = x; k < seed; k+=N) {
					seedSet.insert(k);
					if (nSub1D % 2 == 0) {
						seedSet.insert(k+1);
					}
					// printf("3 - Inserting %d\n", k);
				}	
				// printf("DONE\n");
				partition.seeds.push_back(seedSet);
				seedSet.clear();
			}
			// Create left bridges on left seed DOFs
			if (i == 0) {
				for (int k = y*N; k < seed; k++) {
					seedSet.insert(k);
					if (nSub1D % 2 == 0) {
						seedSet.insert(k+N);
					}
					// printf("4 - Inserting %d\n", k);
				}	
				// printf("DONE\n");
				partition.seeds.push_back(seedSet);
				seedSet.clear();
			}
			// Create lower bridges on bottom seed DOFs
			if (j == nSeedsMid1D-1) {
				for (int k = seed+N; k < N*N; k+=N) {
					seedSet.insert(k);
					if (nSub1D % 2 == 0) {
						seedSet.insert(k+1);
					}
					// printf("5 - Inserting %d\n", k);
				}	
				// printf("DONE\n");
				partition.seeds.push_back(seedSet);
				seedSet.clear();
			}
			// Create right bridges on right seed DOFs
			if (i == nSeedsMid1D-1) {
				for (int k = seed+1; k < (y+1)*N; k++) {
					seedSet.insert(k);
					if (nSub1D % 2 == 0) {
						seedSet.insert(k+N);
					}
					// printf("6 - Inserting %d\n", k);
				}	
				// printf("DONE\n");
				partition.seeds.push_back(seedSet);
				seedSet.clear();
			}
		}
	}
}

void constructSeedsHost(meshPartitionForStage &partition, uint32_t N, uint32_t nSub, uint32_t stageID)
{
	// Create initial seeds vector
	if (stageID == 0) {
    	partition.numSubdomains = nSub * nSub;
		constructSeedsUpperPyramidal(partition, N, nSub);
	}
	else if (stageID == 1) {
		constructSeedsBridge(partition, N, nSub);
		partition.numSubdomains = partition.seeds.size();
	}
	else if (stageID == 2) {
		constructSeedsLowerPyramidal(partition, N, nSub);
    	partition.numSubdomains = partition.seeds.size();
	}
	else if (stageID == 3) {
		constructSeedsDualBridge(partition, N, nSub);
    	partition.numSubdomains = partition.seeds.size();
	}
	
	// Print the seed locations if desired
	// for (int i = 0; i < partition.numSubdomains; i++) {
	//	printf("Subdomain %d\n", i);
	//	for (auto seed : partition.seeds[i]) {
	//		printf("The seed is %d\n", seed);
	//	}	
	// }
}








/* OLD VERSIONS OF SEEDS FUNCTIONS */

/*
void constructSeedsUpperPyramidal(meshPartitionForStage &partition, uint32_t N)
{
	// Initialize seed set
	set<int> seedSet;
	
	// Insert seed 0
	seedSet.insert(0);
	partition.seeds.push_back(seedSet);

	// Insert seed 1
	seedSet.clear();
	seedSet.insert(N-1);
	partition.seeds.push_back(seedSet);

	// Insert seed 2
	seedSet.clear();
	seedSet.insert(N*N-N);
	partition.seeds.push_back(seedSet);

	// Insert seed 3
	seedSet.clear();
	seedSet.insert(N*N-1);
	partition.seeds.push_back(seedSet);
}
*/

/*	
void constructSeedsBridge(meshPartitionForStage &partition, uint32_t N)
{
	// Initialize seed set
	set<int> seedSet;

	// Seeds
	
	// 0 - Bottom	
	for (int i = 1; i < N-1; i++) {
		seedSet.insert(i);
	}
	partition.seeds.push_back(seedSet);
	
	// 1 - Left
	seedSet.clear();
	for (int i = N; i < N*N-N; i += N) {
		seedSet.insert(i);
	}
	partition.seeds.push_back(seedSet);

	// 2 - Right
	seedSet.clear();
	for (int i = 2*N-1; i < N*N-1; i += N) {
		seedSet.insert(i);
	}
	partition.seeds.push_back(seedSet);

	// 3 - Top	
	seedSet.clear();
	for (int i = N*N-N+1; i < N*N-1; i++) {
		seedSet.insert(i);
	}
	partition.seeds.push_back(seedSet);
}
*/

/*
void constructSeedsLowerPyramidal(meshPartitionForStage &partition, uint32_t N)
{
	// Initialize seed set
	set<int> seedSet;

	// Seeds
	uint32_t bottomLeft = (N/2-1) * N + (N/2-1);
	uint32_t bottomRight = bottomLeft + 1;
	uint32_t topLeft = bottomLeft + N;
	uint32_t topRight = bottomLeft + N + 1;

	// Add seeds to set
	seedSet.insert(bottomLeft);
	seedSet.insert(bottomRight);
	seedSet.insert(topLeft);
	seedSet.insert(topRight);

	// Add set to vector
	partition.seeds.push_back(seedSet);
}
*/

/*
void constructSeedsDualBridge(meshPartitionForStage &partition, uint32_t N)
{
	// Initialize seed set
	set<int> seedSet;

	// Seeds
	uint32_t bottom1, bottom2, left1, left2, right1, right2, top1, top2;
	
	// Bottom side seeds
	for (int i = 0; i < N/2-1; i++) {
		bottom1 = (N/2 - 1) + i * N;
		bottom2 = (N/2) + i * N;
		seedSet.insert(bottom1);
		seedSet.insert(bottom2);
	}
	partition.seeds.push_back(seedSet);

	// Left side seeds
	seedSet.clear();
	for (int i = 0; i < N/2-1; i++) {
		left1 = (N/2 - 1) * N + i;
		left2 = (N/2) * N + i;
		seedSet.insert(left1);
		seedSet.insert(left2);
	}
	partition.seeds.push_back(seedSet);
		
	// Right side seeds
	seedSet.clear();
	for (int i = 0; i < N/2-1; i++) {
		right1 = (N/2 + 1) * N - 1 - i;
		right2 = (N/2) * N - 1 - i;
		seedSet.insert(right1);
		seedSet.insert(right2);
	}
	partition.seeds.push_back(seedSet);

	// Top side seeds
	seedSet.clear();
	for (int i = 0; i < N/2-1; i++) {
		top1 = (N - 1 - i) * N + N/2 - 1;
		top2 = (N - 1 - i) * N + N/2;
		seedSet.insert(top1);
		seedSet.insert(top2);
	}
	partition.seeds.push_back(seedSet);
}
*/
