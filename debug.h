void printDeviceSimilarity(float * solutionSwept_d, float * solutionGlobal_d, float TOL, uint32_t Ndofs, uint32_t N)
{
	float * solutionSwept = new float[Ndofs];
	float * solutionGlobal = new float[Ndofs];
	cudaMemcpy(solutionSwept, solutionSwept_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	cudaMemcpy(solutionGlobal, solutionGlobal_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			if (abs(solutionSwept[dof] - solutionGlobal[dof]) < TOL) {
				printf("1 ");
			}
			else {
				printf("0 ");
			}
		}
		printf("\n");
	}	
}

void printDeviceSimilarity1D(float * solutionSwept_d, float * solutionGlobal_d, float TOL, uint32_t Ndofs)
{
	float * solutionSwept = new float[Ndofs];
	float * solutionGlobal = new float[Ndofs];
	cudaMemcpy(solutionSwept, solutionSwept_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	cudaMemcpy(solutionGlobal, solutionGlobal_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < Ndofs; i++) {
		uint32_t dof = i;
		if (abs(solutionSwept[dof] - solutionGlobal[dof]) < TOL) {
			printf("1 ");
		}
		else {
			printf("0 ");
		}
	}
}

void printDeviceSolutionComparison(float * solutionSwept_d, float * solutionGlobal_d, uint32_t Ndofs)
{
	float * solutionSwept = new float[Ndofs];
	float * solutionGlobal = new float[Ndofs];
	cudaMemcpy(solutionSwept, solutionSwept_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	cudaMemcpy(solutionGlobal, solutionGlobal_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < Ndofs; i++) {
		uint32_t dof = i;
		printf("%f, %f\n", solutionGlobal[dof], solutionSwept[dof]); 
	}
}

void printDeviceDifference(float * solutionSwept_d, float * solutionGlobal_d, float TOL, uint32_t Ndofs, uint32_t N)
{
	float maxDifference = 0;
	uint32_t maxI = 0;
	uint32_t maxJ = 0;
	float * solutionSwept = new float[Ndofs];
	float * solutionGlobal = new float[Ndofs];
	cudaMemcpy(solutionSwept, solutionSwept_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	cudaMemcpy(solutionGlobal, solutionGlobal_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < Ndofs; i++) {
		uint32_t iX = i % N;
		uint32_t iY = i / N;
		uint32_t blockX = iX / 16;
		uint32_t blockY = iY / 16;
		uint32_t blockID = blockX + blockY * (N/16);
		float difference = abs(solutionSwept[i] - solutionGlobal[i]);
		printf("i = %d, j = %d, Block_x = %d, Block_y = %d: Diff[%d] (Block %d) = %.15f\n", iX, iY, blockX, blockY, i, blockID, abs(solutionSwept[i] - solutionGlobal[i]));
		printf("\n");
		if (difference > maxDifference) {
			maxDifference = difference;
			maxI = iX;	
			maxJ = iY;	
		}
	}
	printf("The max difference is %.15f at (iX, iY) = (%d, %d)\n", maxDifference, maxI, maxJ);	
}

void printDeviceSolution(float * solutionSwept_d, uint32_t Ndofs)
{
	float * solutionSwept = new float[Ndofs];
	cudaMemcpy(solutionSwept, solutionSwept_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < Ndofs; i++) {
		printf("solution[%d] = %f\n", i, solutionSwept[i]);
	}
}

void printHostSolution(float * solutionSwept, uint32_t Ndofs)
{
	for (int i = 0; i < Ndofs; i++) {
		printf("solution[%d] = %f\n", i, solutionSwept[i]);
	}
}


void printPartitionInformation(meshPartitionForStage partition, linearSystem matrix, float * solution, uint32_t * iterationLevel, uint32_t N)
{
	// Subdomain of DOFs
	printf("Subdomains\n");
	printHostSolutionInt(partition.subdomainOfDOFs, matrix.Ndofs, N);

	// Current Solution
	// printf("Solution\n");
	// printDeviceSolutionFloat(solution, matrix.Ndofs, N);

	// Iteration Level
	printf("Iteration Level\n");
	printDeviceSolutionInt(iterationLevel, matrix.Ndofs, N);
}

void determineIfValidIterationLevel(uint32_t * iterationLevel, linearSystem matrix) 
{
	// Determine if the iteration levels are valid (neighbors are no more than one iteration apart)
	bool validIterationLevel = true;
	for (int dof = 0; dof < matrix.Ndofs; dof++) {
		uint32_t dofIterLevel = iterationLevel[dof];
		for (int idx = matrix.indexPtr[dof]; idx < matrix.indexPtr[dof+1]; idx++) {
			uint32_t n = matrix.nodeNeighbors[idx];
			uint32_t neighborIterLevel = iterationLevel[n];
			int delta = dofIterLevel - neighborIterLevel;
			if (abs(delta) > 1) {
				validIterationLevel = false;
			}
		}
	}	

	// Print the result
	if (validIterationLevel == true) {
		printf("Valid Iteration Level");
	}
	else {
		printf("Not Valid Iteration Level");
	}

}

 
void printGlobalAndSharedMatchDevice(float * solution_d, float * solutionGM, uint32_t * iterationLevel_d, uint32_t numIterations, uint32_t Ndofs)
{
	float * solutionShared = new float[Ndofs];
	float * solutionGlobal = new float[Ndofs];
	uint32_t * iterationLevel = new uint32_t[Ndofs];
	cudaMemcpy(solutionShared, solution_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);	
	cudaMemcpy(solutionGlobal, solutionGM, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);	
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	

	bool correctBool = true;
	for (int i = 0; i < Ndofs; i++) {
		if (iterationLevel[i] == numIterations) {
			// printf("\nCheck @ dof %d", i);
			if (abs(solutionGlobal[i] - solutionShared[i]) > 1e-6) {
				correctBool = false;
				printf("Disparity at dof = %d: global = %f, shared = %f\n", i, solutionGlobal[i], solutionShared[i]);
			}
		}	
	}

	printf("\nThe global and shared match for all DOFs at iteration %d: %d\n", numIterations, correctBool);
}
