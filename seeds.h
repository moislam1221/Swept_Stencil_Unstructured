void constructSeedsUpperPyramidal(vector<set<int>> &seeds, uint32_t numSubdomains, uint32_t N)
{
	// Initialize seed set
	set<int> seedSet;

	// Insert seed 0
	seedSet.insert(0);
	seeds.push_back(seedSet);

	// Insert seed 1
	seedSet.clear();
	seedSet.insert(N-1);
	seeds.push_back(seedSet);

	// Insert seed 2
	seedSet.clear();
	seedSet.insert(N*N-N);
	seeds.push_back(seedSet);

	// Insert seed 3
	seedSet.clear();
	seedSet.insert(N*N-1);
	seeds.push_back(seedSet);

}

void constructSeedsBridge(vector<set<int>> &seeds, uint32_t numSubdomains, uint32_t N)
{
	// Initialize seed set
	set<int> seedSet;

	// Seeds
	
	// 0 - Bottom	
	for (int i = 1; i < N-1; i++) {
		seedSet.insert(i);
	}
	seeds.push_back(seedSet);
	
	// 1 - Left
	seedSet.clear();
	for (int i = N; i < N*N-N; i += N) {
		seedSet.insert(i);
	}
	seeds.push_back(seedSet);

	// 2 - Right
	seedSet.clear();
	for (int i = 2*N-1; i < N*N-1; i += N) {
		seedSet.insert(i);
	}
	seeds.push_back(seedSet);

	// 3 - Top	
	seedSet.clear();
	for (int i = N*N-N+1; i < N*N-1; i++) {
		seedSet.insert(i);
	}
	seeds.push_back(seedSet);
}

void constructSeedsLowerPyramidal(vector<set<int>> &seeds, uint32_t numSubdomains, uint32_t N)
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
	seeds.push_back(seedSet);
}


void constructSeedsDualBridge(vector<set<int>> &seeds, uint32_t numSubdomains, uint32_t N)
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
	seeds.push_back(seedSet);

	// Left side seeds
	seedSet.clear();
	for (int i = 0; i < N/2-1; i++) {
		left1 = (N/2 - 1) * N + i;
		left2 = (N/2) * N + i;
		seedSet.insert(left1);
		seedSet.insert(left2);
	}
	seeds.push_back(seedSet);
		
	// Right side seeds
	seedSet.clear();
	for (int i = 0; i < N/2-1; i++) {
		right1 = (N/2 + 1) * N - 1 - i;
		right2 = (N/2) * N - 1 - i;
		seedSet.insert(right1);
		seedSet.insert(right2);
	}
	seeds.push_back(seedSet);

	// Top side seeds
	seedSet.clear();
	for (int i = 0; i < N/2-1; i++) {
		top1 = (N - 1 - i) * N + N/2 - 1;
		top2 = (N - 1 - i) * N + N/2;
		seedSet.insert(top1);
		seedSet.insert(top2);
	}
	seeds.push_back(seedSet);

}
