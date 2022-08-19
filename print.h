// Print from CPU Host

// Int
void printHostSolutionInt(uint32_t * solution, uint32_t Ndofs, uint32_t N)
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%d ", solution[dof]);
		}
		printf("\n");
	}	
}

// Float
void printHostSolutionFloat(float * solution, uint32_t Ndofs, uint32_t N)
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%f ", solution[dof]);
		}
		printf("\n");
	}	
}

// Flattened Int Array
void printHostArrayInt(uint32_t * solution, uint32_t Ndofs)
{
	for (int i = 0; i < Ndofs; i++) {
		printf("%d ", solution[i]);
	}	
}

// Print from GPU Device

// Int
void printDeviceSolutionInt(uint32_t * solution_d, uint32_t Ndofs, uint32_t N)
{
	uint32_t * solution = new uint32_t[Ndofs];
	cudaMemcpy(solution, solution_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%d ", solution[dof]);
		}
		printf("\n");
	}	
}

// Float
void printDeviceSolutionFloat(float * solution_d, uint32_t Ndofs, uint32_t N)
{
	float * solution = new float[Ndofs];
	cudaMemcpy(solution, solution_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%f ", solution[dof]);
		}
		printf("\n");
	}	
}

// Flattened Int Array
void printDeviceArrayInt(uint32_t * solution_d, uint32_t Ndofs)
{
	uint32_t * solution = new uint32_t[Ndofs];
	cudaMemcpy(solution, solution_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < Ndofs; i++) {
		printf("%d\n ", solution[i]);
	}	
}


// Flattened Int Array
void printDeviceArrayFloat(float * solution_d, uint32_t Ndofs)
{
	float * solution = new float[Ndofs];
	cudaMemcpy(solution, solution_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < Ndofs; i++) {
		printf("%f\n", solution[i]);
	}	
}


