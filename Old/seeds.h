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
	uint32_t seedX, seedY, seedID, shift;
	for (int j = 0; j < nSub1D; j++) {
		for (int i = 0; i < nSub1D; i++) {
			seedX = seeds1D[i];
			seedY = seeds1D[j];
			seedID = seedX + N * seedY;
			seedSet.insert(seedID);
			shift = 0;
			// printf("i = %d, j = %d, seedID = %d\n", i, j, seedID);
			// If each subdomain has even dimension, interior subdomain seeds are 2 by 2
			if (i > 0 && i < nSub1D-1 && j > 0 && j < nSub1D-1 && nPerSub % 2 == 0) {
				seedSet.insert(seedID+1);
				seedSet.insert(seedID+N);
				seedSet.insert(seedID+N+1);
			}
			// If not the corners
			else if ((i != 0 || j!= 0) && (i != 0 || j != nSub1D-1) && (i != nSub1D-1 || j!= 0) && (i != nSub1D-1 || j != nSub1D-1)) {
				for (int idx = 0; idx <= (int)ceil((float)(nPerSub-1)/2); idx++) {
					// Left column
					if (i == 0 && j != 0 && j!= nSub1D-1) {
						shift = idx;
					}
					// Right column
					else if (i == nSub1D-1 && j != 0 && j!= nSub1D-1) {
						shift = -idx;
					}
					// Top column
					else if (j == 0 && i != 0 && i != nSub1D-1) {
						shift = N * idx;
					}
					// Bottom column
					else if (j == nSub1D-1 && i != 0 && i != nSub1D-1) {
						shift = -N * idx;
					}
					seedSet.insert(seedID + shift);
					// Add extra row/column of seeds in nPerSub is even
					if (nPerSub % 2 == 0) {
						if (i == 0 || i == nSub1D-1) {
							seedSet.insert(seedID + shift + N);
						}
						else if (j == 0 || j == nSub1D-1) {
							seedSet.insert(seedID + shift + 1);
						}
					}
					// printf("seedX = %d, seedY = %d, seedID = %d\n", seedX, seedY, seedID+1);
				}
			}
			// Seeds on the corners
			else {
				for (int idy = 0; idy <= (int)ceil((float)(nPerSub-1)/2); idy++) {
					for (int idx = 0; idx <= ceil((float)(nPerSub-1)/2); idx++) {
						// Top-Left corner 
						if (i == 0 && j == 0) {
							shift = idx + N * idy;
						}
						// Top-Right corner 
						else if (i == nSub1D-1 && j == 0) {
							shift = -idx + N * idy;
						}
						// Bottom-Left corner 
						else if (i == 0 && j == nSub1D-1) {
							shift = idx - N * idy;
						}
						// Bottom-Right corner 
						else if (i == nSub1D-1 && j == nSub1D-1) {
							shift = -idx - N * idy;
						}
						seedSet.insert(seedID + shift);
						// printf("seedX = %d, seedY = %d, seedID = %d\n", seedX, seedY, seedID+1);
					}
				}
			}
			partition.seeds.push_back(seedSet);
			seedSet.clear();
		}
	}

}

void constructSeedsUpperPyramidalNew(meshPartitionForStageNew &partition, uint32_t N, uint32_t nSub1D)
{
	// nPerSub
	uint32_t nPerSub = N / nSub1D;

	// Initialize vector containing indices in 1D direction of seeds
	vector<int> seeds1D;
	fillInSeeds1D(seeds1D, N, nSub1D);
	
	// Fill in partition.seeds with all seed IDs (1D => 2D but with a flattened 1D index)
	// Initialize seed set variable
	set<int> seedSet;
	uint32_t seedX, seedY, seedID, shift;
	for (int j = 0; j < nSub1D; j++) {
		for (int i = 0; i < nSub1D; i++) {
			seedX = seeds1D[i];
			seedY = seeds1D[j];
			seedID = seedX + N * seedY;
			seedSet.insert(seedID);
			shift = 0;
			// printf("i = %d, j = %d, seedID = %d\n", i, j, seedID);
			// If each subdomain has even dimension, interior subdomain seeds are 2 by 2
			if (i > 0 && i < nSub1D-1 && j > 0 && j < nSub1D-1 && nPerSub % 2 == 0) {
				seedSet.insert(seedID+1);
				seedSet.insert(seedID+N);
				seedSet.insert(seedID+N+1);
			}
			// If not the corners
			else if ((i != 0 || j!= 0) && (i != 0 || j != nSub1D-1) && (i != nSub1D-1 || j!= 0) && (i != nSub1D-1 || j != nSub1D-1)) {
				for (int idx = 0; idx <= (int)ceil((float)(nPerSub-1)/2); idx++) {
					// Left column
					if (i == 0 && j != 0 && j!= nSub1D-1) {
						shift = idx;
					}
					// Right column
					else if (i == nSub1D-1 && j != 0 && j!= nSub1D-1) {
						shift = -idx;
					}
					// Top column
					else if (j == 0 && i != 0 && i != nSub1D-1) {
						shift = N * idx;
					}
					// Bottom column
					else if (j == nSub1D-1 && i != 0 && i != nSub1D-1) {
						shift = -N * idx;
					}
					seedSet.insert(seedID + shift);
					// Add extra row/column of seeds in nPerSub is even
					if (nPerSub % 2 == 0) {
						if (i == 0 || i == nSub1D-1) {
							seedSet.insert(seedID + shift + N);
						}
						else if (j == 0 || j == nSub1D-1) {
							seedSet.insert(seedID + shift + 1);
						}
					}
					// printf("seedX = %d, seedY = %d, seedID = %d\n", seedX, seedY, seedID+1);
				}
			}
			// Seeds on the corners
			else {
				for (int idy = 0; idy <= (int)ceil((float)(nPerSub-1)/2); idy++) {
					for (int idx = 0; idx <= ceil((float)(nPerSub-1)/2); idx++) {
						// Top-Left corner 
						if (i == 0 && j == 0) {
							shift = idx + N * idy;
						}
						// Top-Right corner 
						else if (i == nSub1D-1 && j == 0) {
							shift = -idx + N * idy;
						}
						// Bottom-Left corner 
						else if (i == 0 && j == nSub1D-1) {
							shift = idx - N * idy;
						}
						// Bottom-Right corner 
						else if (i == nSub1D-1 && j == nSub1D-1) {
							shift = -idx - N * idy;
						}
						seedSet.insert(seedID + shift);
						// printf("seedX = %d, seedY = %d, seedID = %d\n", seedX, seedY, seedID+1);
					}
				}
			}
			partition.seeds.push_back(seedSet);
			seedSet.clear();
		}
	}

}

void constructSeedsBridge(meshPartitionForStage &partition, uint32_t N, uint32_t nSub1D)
{
	// nPerSub
	uint32_t nPerSub = ceil(N / nSub1D);
	
	// Initialize vector containing indices in 1D direction of seeds
	vector<int> seeds1D;
	fillInSeeds1D(seeds1D, N, nSub1D);
	
	// Fill in partition.seeds with all seed IDs (1D => 2D but with a flattened 1D index)
	vector<int> seeds2D;
	constructSeeds2DFrom1D(seeds2D, seeds1D, N);

	// Initialize seed set
	set<int> seedSet;
	for (int i = 0; i < nSub1D * nSub1D; i++) {
		uint32_t shift = 0;
		// Horizontal connector seeds
		if (i % nSub1D != nSub1D-1) {
			uint32_t seedBegin = seeds2D[i]; 
			uint32_t seedEnd = seeds2D[i+1];
			if (i % nSub1D == 0) {
				seedBegin += (int)ceil((float)(nPerSub-1) / 2);
			}
			else if (i % nSub1D == nSub1D-2) {
				seedEnd -= (int)ceil((float)(nPerSub-1) / 2);
			}
			if (i < nSub1D || i > nSub1D * (nSub1D-1) - 1) {
				for (int j = seedBegin+1; j < seedEnd; j++) {
					for (int idx_row = 0; idx_row <= (int)ceil((float)(nPerSub-1)/2); idx_row++) {
						if (i < nSub1D) {
							shift = idx_row * N;
						}
						else if (i > nSub1D * (nSub1D-1) -1) {
							shift = -idx_row * N;
						}
						seedSet.insert(j + shift);
					}
				}
			}
			else {
				for (int j = seedBegin+1; j < seedEnd; j++) {
					seedSet.insert(j);
					if (nPerSub % 2 == 0) {
						seedSet.insert(j + N);
					}
				}
			}
			partition.seeds.push_back(seedSet);
			seedSet.clear();
		}
		// Vertical connector seeds
		if (i < nSub1D * (nSub1D-1)) {
			uint32_t seedBegin = seeds2D[i]; 
			uint32_t seedEnd = seeds2D[i+nSub1D]; 
			uint32_t shift = 0;
			if (i < nSub1D) {
				seedBegin += (int)ceil((float)(nPerSub-1) / 2) * N;
			}
			else if (i > nSub1D * (nSub1D-2) - 1) {
				seedEnd -= (int)ceil((float)(nPerSub-1) / 2) * N;
			}
			if (i % nSub1D == 0 || i % nSub1D == nSub1D-1) {
				for (int j = seedBegin+N; j < seedEnd; j+=N) {
					for (int idx_col = 0; idx_col <= (int)ceil((float)(nPerSub-1)/2); idx_col++) {
						if (i % nSub1D == 0) {
							shift = idx_col;
						}
						else if (i % nSub1D == nSub1D-1) {
							shift = -idx_col;
						}
						seedSet.insert(j + shift);
					}
				}
			}
			// printf("(%d, %d)\n", seedBegin, seedEnd);
			else {
				for (int j = seedBegin+N; j < seedEnd; j += N) {
					// printf("%d\n", j);
					seedSet.insert(j);
					if (nPerSub % 2 == 0) {
						seedSet.insert(j + 1);
					}
				}

			}	
			partition.seeds.push_back(seedSet);
			seedSet.clear();
		}
	}
}

void constructSeedsBridgeNew(meshPartitionForStageNew &partition, uint32_t N, uint32_t nSub1D)
{
	// nPerSub
	uint32_t nPerSub = ceil(N / nSub1D);
	
	// Initialize vector containing indices in 1D direction of seeds
	vector<int> seeds1D;
	fillInSeeds1D(seeds1D, N, nSub1D);
	
	// Fill in partition.seeds with all seed IDs (1D => 2D but with a flattened 1D index)
	vector<int> seeds2D;
	constructSeeds2DFrom1D(seeds2D, seeds1D, N);

	// Initialize seed set
	set<int> seedSet;
	for (int i = 0; i < nSub1D * nSub1D; i++) {
		uint32_t shift = 0;
		// Horizontal connector seeds
		if (i % nSub1D != nSub1D-1) {
			uint32_t seedBegin = seeds2D[i]; 
			uint32_t seedEnd = seeds2D[i+1];
			if (i % nSub1D == 0) {
				seedBegin += (int)ceil((float)(nPerSub-1) / 2);
			}
			else if (i % nSub1D == nSub1D-2) {
				seedEnd -= (int)ceil((float)(nPerSub-1) / 2);
			}
			if (i < nSub1D || i > nSub1D * (nSub1D-1) - 1) {
				for (int j = seedBegin+1; j < seedEnd; j++) {
					for (int idx_row = 0; idx_row <= (int)ceil((float)(nPerSub-1)/2); idx_row++) {
						if (i < nSub1D) {
							shift = idx_row * N;
						}
						else if (i > nSub1D * (nSub1D-1) -1) {
							shift = -idx_row * N;
						}
						seedSet.insert(j + shift);
					}
				}
			}
			else {
				for (int j = seedBegin+1; j < seedEnd; j++) {
					seedSet.insert(j);
					if (nPerSub % 2 == 0) {
						seedSet.insert(j + N);
					}
				}
			}
			partition.seeds.push_back(seedSet);
			seedSet.clear();
		}
		// Vertical connector seeds
		if (i < nSub1D * (nSub1D-1)) {
			uint32_t seedBegin = seeds2D[i]; 
			uint32_t seedEnd = seeds2D[i+nSub1D]; 
			uint32_t shift = 0;
			if (i < nSub1D) {
				seedBegin += (int)ceil((float)(nPerSub-1) / 2) * N;
			}
			else if (i > nSub1D * (nSub1D-2) - 1) {
				seedEnd -= (int)ceil((float)(nPerSub-1) / 2) * N;
			}
			if (i % nSub1D == 0 || i % nSub1D == nSub1D-1) {
				for (int j = seedBegin+N; j < seedEnd; j+=N) {
					for (int idx_col = 0; idx_col <= (int)ceil((float)(nPerSub-1)/2); idx_col++) {
						if (i % nSub1D == 0) {
							shift = idx_col;
						}
						else if (i % nSub1D == nSub1D-1) {
							shift = -idx_col;
						}
						seedSet.insert(j + shift);
					}
				}
			}
			// printf("(%d, %d)\n", seedBegin, seedEnd);
			else {
				for (int j = seedBegin+N; j < seedEnd; j += N) {
					// printf("%d\n", j);
					seedSet.insert(j);
					if (nPerSub % 2 == 0) {
						seedSet.insert(j + 1);
					}
				}

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
	vector<int> seedsDual1D;
	uint32_t seedID;
	for (int i = 0; i < nSub1D-1; i++) {
		seedID = (nPerSub-1) + i * nPerSub; 
		seedsDual1D.push_back(seedID);
		// printf("i = %d, seedID = %d\n", i, seeds1DMid[i]);
	}

	// Fill in partition.seeds with all seed IDs (1D => 2D but with a flattened 1D index)
	// Initialize seed set variable
	set<int> seedSet;
	uint32_t seedX, seedY;
	for (int j = 0; j < nSub1D-1; j++) {
		for (int i = 0; i < nSub1D-1; i++) {
			seedX = seedsDual1D[i];
			seedY = seedsDual1D[j];
			seedID = seedX + N * seedY;
			// Gte 2 by 2 dual seed given topleft seedID
			seedSet.insert(seedID);
			seedSet.insert(seedID+1);
			seedSet.insert(seedID+N);
			seedSet.insert(seedID+N+1);
			// Add 2 by 2 dual seed to set of seeds
			partition.seeds.push_back(seedSet);
			seedSet.clear();
		}
	}
}

void constructSeedsLowerPyramidalNew(meshPartitionForStageNew &partition, uint32_t N, uint32_t nSub1D)
{
	// nPerSub
	uint32_t nPerSub = N / nSub1D;
	
	// Initialize vector containing indices in 1D direction of seeds
	vector<int> seedsDual1D;
	uint32_t seedID;
	for (int i = 0; i < nSub1D-1; i++) {
		seedID = (nPerSub-1) + i * nPerSub; 
		seedsDual1D.push_back(seedID);
		// printf("i = %d, seedID = %d\n", i, seeds1DMid[i]);
	}

	// Fill in partition.seeds with all seed IDs (1D => 2D but with a flattened 1D index)
	// Initialize seed set variable
	set<int> seedSet;
	uint32_t seedX, seedY;
	for (int j = 0; j < nSub1D-1; j++) {
		for (int i = 0; i < nSub1D-1; i++) {
			seedX = seedsDual1D[i];
			seedY = seedsDual1D[j];
			seedID = seedX + N * seedY;
			// Gte 2 by 2 dual seed given topleft seedID
			seedSet.insert(seedID);
			seedSet.insert(seedID+1);
			seedSet.insert(seedID+N);
			seedSet.insert(seedID+N+1);
			// Add 2 by 2 dual seed to set of seeds
			partition.seeds.push_back(seedSet);
			seedSet.clear();
		}
	}
}

void constructSeedsDualBridge(meshPartitionForStage &partition, uint32_t N, uint32_t nSub1D)
{
	// nPerSub
	uint32_t nPerSub = N / nSub1D;
	
	// Initialize vector containing indices in 1D direction of seeds
	vector<int> seedsDual1D;
	uint32_t seedID;
	for (int i = 0; i < nSub1D-1; i++) {
		seedID = (nPerSub-1) + i * nPerSub; 
		seedsDual1D.push_back(seedID);
		// printf("i = %d, seedID = %d\n", i, seeds1DMid[i]);
	}

	// Fill in partition.seeds with all seed IDs (1D => 2D but with a flattened 1D index)
	vector<int> seedsDual2D;
	constructSeeds2DFrom1D(seedsDual2D, seedsDual1D, N);

	// Construct dual bridges
	set<int> seedSet;
	uint32_t nSeeds1D = seedsDual1D.size();
	uint32_t seedBeginID, seedEndID, seedBegin, seedEnd, seedBeginRef;
	for (int j = 0; j < nSeeds1D; j++) {
		for (int i = 0; i < nSeeds1D; i++) {
			seedBeginID = i + j * nSeeds1D;
			seedBeginRef = seedsDual2D[seedBeginID];
			// Slight adjustment to beginning seed for 2 by 2
			seedBegin = seedBeginRef+1;
			// Create horizontal bridges between seeds
			if (i != nSeeds1D - 1) {
				seedEndID = (i+1) + j * nSeeds1D;
				seedEnd = seedsDual2D[seedEndID];
				// printf("seedBegin = %d, seedEnd = %d\n", seedBegin, seedEnd);
				for (int k = seedBegin+1; k < seedEnd; k++) {
					seedSet.insert(k);
					seedSet.insert(k+N);
					// printf("1 - Inserting %d\n", k);
				}
				// printf("DONE\n");
				partition.seeds.push_back(seedSet);
				seedSet.clear();
			}
			// Slight adjustment to beginning seed for 2 by 2
			seedBegin = seedBeginRef + N;
			// Create vertical bridges between seeds
			if (j != nSeeds1D - 1) {
				seedEndID = i + (j+1) * nSeeds1D;
				seedEnd = seedsDual2D[seedEndID];
				for (int k = seedBegin+N; k < seedEnd; k+=N) {
					seedSet.insert(k);
					seedSet.insert(k+1);
					// printf("2 - Inserting %d\n", k);
				}
				// printf("DONE\n");
				partition.seeds.push_back(seedSet);
				seedSet.clear();
			}
		}
	}

	uint32_t seed, seed2D, x, y;	
	for (int j = 0; j < nSeeds1D; j++) {
		for (int i = 0; i < nSeeds1D; i++) {
			// Determine the seed location and x and y coordinates
			seedID = i + j * nSeeds1D;
			seed = seedsDual2D[seedID];
			x = seed % N;
			y = seed / N;
			// Create upper bridges on top seed DOFs
			if (j == 0) {
				for (int k = x; k < seed; k+=N) {
					seedSet.insert(k);
					seedSet.insert(k+1);
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
					seedSet.insert(k+N);
					// printf("4 - Inserting %d\n", k);
				}	
				// printf("DONE\n");
				partition.seeds.push_back(seedSet);
				seedSet.clear();
			}
			// Create lower bridges on bottom seed DOFs
			if (j == nSeeds1D-1) {
				for (int k = seed+N; k < N*N; k+=N) {
					seedSet.insert(k);
					seedSet.insert(k+1);
					// printf("5 - Inserting %d\n", k);
				}	
				// printf("DONE\n");
				partition.seeds.push_back(seedSet);
				seedSet.clear();
			}
			// Create right bridges on right seed DOFs
			if (i == nSeeds1D-1) {
				for (int k = seed+1; k < (y+1)*N; k++) {
					seedSet.insert(k);
					seedSet.insert(k+N);
					// printf("6 - Inserting %d\n", k);
				}	
				// printf("DONE\n");
				partition.seeds.push_back(seedSet);
				seedSet.clear();
			}
		}
	}

}

void constructSeedsDualBridgeNew(meshPartitionForStageNew &partition, uint32_t N, uint32_t nSub1D)
{
	// nPerSub
	uint32_t nPerSub = N / nSub1D;
	
	// Initialize vector containing indices in 1D direction of seeds
	vector<int> seedsDual1D;
	uint32_t seedID;
	for (int i = 0; i < nSub1D-1; i++) {
		seedID = (nPerSub-1) + i * nPerSub; 
		seedsDual1D.push_back(seedID);
		// printf("i = %d, seedID = %d\n", i, seeds1DMid[i]);
	}

	// Fill in partition.seeds with all seed IDs (1D => 2D but with a flattened 1D index)
	vector<int> seedsDual2D;
	constructSeeds2DFrom1D(seedsDual2D, seedsDual1D, N);

	// Construct dual bridges
	set<int> seedSet;
	uint32_t nSeeds1D = seedsDual1D.size();
	uint32_t seedBeginID, seedEndID, seedBegin, seedEnd, seedBeginRef;
	for (int j = 0; j < nSeeds1D; j++) {
		for (int i = 0; i < nSeeds1D; i++) {
			seedBeginID = i + j * nSeeds1D;
			seedBeginRef = seedsDual2D[seedBeginID];
			// Slight adjustment to beginning seed for 2 by 2
			seedBegin = seedBeginRef+1;
			// Create horizontal bridges between seeds
			if (i != nSeeds1D - 1) {
				seedEndID = (i+1) + j * nSeeds1D;
				seedEnd = seedsDual2D[seedEndID];
				// printf("seedBegin = %d, seedEnd = %d\n", seedBegin, seedEnd);
				for (int k = seedBegin+1; k < seedEnd; k++) {
					seedSet.insert(k);
					seedSet.insert(k+N);
					// printf("1 - Inserting %d\n", k);
				}
				// printf("DONE\n");
				partition.seeds.push_back(seedSet);
				seedSet.clear();
			}
			// Slight adjustment to beginning seed for 2 by 2
			seedBegin = seedBeginRef + N;
			// Create vertical bridges between seeds
			if (j != nSeeds1D - 1) {
				seedEndID = i + (j+1) * nSeeds1D;
				seedEnd = seedsDual2D[seedEndID];
				for (int k = seedBegin+N; k < seedEnd; k+=N) {
					seedSet.insert(k);
					seedSet.insert(k+1);
					// printf("2 - Inserting %d\n", k);
				}
				// printf("DONE\n");
				partition.seeds.push_back(seedSet);
				seedSet.clear();
			}
		}
	}

	uint32_t seed, seed2D, x, y;	
	for (int j = 0; j < nSeeds1D; j++) {
		for (int i = 0; i < nSeeds1D; i++) {
			// Determine the seed location and x and y coordinates
			seedID = i + j * nSeeds1D;
			seed = seedsDual2D[seedID];
			x = seed % N;
			y = seed / N;
			// Create upper bridges on top seed DOFs
			if (j == 0) {
				for (int k = x; k < seed; k+=N) {
					seedSet.insert(k);
					seedSet.insert(k+1);
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
					seedSet.insert(k+N);
					// printf("4 - Inserting %d\n", k);
				}	
				// printf("DONE\n");
				partition.seeds.push_back(seedSet);
				seedSet.clear();
			}
			// Create lower bridges on bottom seed DOFs
			if (j == nSeeds1D-1) {
				for (int k = seed+N; k < N*N; k+=N) {
					seedSet.insert(k);
					seedSet.insert(k+1);
					// printf("5 - Inserting %d\n", k);
				}	
				// printf("DONE\n");
				partition.seeds.push_back(seedSet);
				seedSet.clear();
			}
			// Create right bridges on right seed DOFs
			if (i == nSeeds1D-1) {
				for (int k = seed+1; k < (y+1)*N; k++) {
					seedSet.insert(k);
					seedSet.insert(k+N);
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

void constructSeedsHostNew(meshPartitionForStageNew &partition, uint32_t N, uint32_t nSub, uint32_t stageID)
{
	// Create initial seeds vector
	if (stageID == 0) {
    	partition.numSubdomains = nSub * nSub;
		constructSeedsUpperPyramidalNew(partition, N, nSub);
	}
	else if (stageID == 1) {
		constructSeedsBridgeNew(partition, N, nSub);
		partition.numSubdomains = partition.seeds.size();
	}
	else if (stageID == 2) {
		constructSeedsLowerPyramidalNew(partition, N, nSub);
    	partition.numSubdomains = partition.seeds.size();
	}
	else if (stageID == 3) {
		constructSeedsDualBridgeNew(partition, N, nSub);
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
