uint32_t determineNumJacobiIters(uint32_t * iterationLevel, uint32_t Ndofs)
{
	uint32_t minLevel = UINT32_MAX;
	uint32_t maxLevel = 0;
	for (int i = 0; i < Ndofs; i++) {
		uint32_t level = iterationLevel[i];
		if (level < minLevel) {
			minLevel = level;
		}
		if (level > maxLevel) {
			maxLevel = level;
		}
	}
	return maxLevel - minLevel;
}

uint32_t determineMaxLevel(uint32_t * iterationLevel, uint32_t Ndofs)
{
	uint32_t maxLevel = 0;
	for (int i = 0; i < Ndofs; i++) {
		uint32_t level = iterationLevel[i];
		if (level > maxLevel) {
			maxLevel = level;
		}
	}
	return maxLevel;
}

uint32_t determineMinLevel(uint32_t * iterationLevel, uint32_t Ndofs)
{
	uint32_t minLevel = UINT32_MAX;
	for (int i = 0; i < Ndofs; i++) {
		uint32_t level = iterationLevel[i];
		if (level < minLevel) {
			minLevel = level;
		}
	}
	return minLevel;
}

__global__
void copySolutionBuffers(float * evenSolutionBuffer, float * oddSolutionBuffer, uint32_t *iterationLevel, float * evenSolutionBufferOutput, float * oddSolutionBufferOutput, uint32_t * iterationLevelOutput, uint32_t Ndofs)
{
	uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t stride = blockDim.x * gridDim.x;
	for (int i = index; i < Ndofs; i += stride) {
		evenSolutionBuffer[i] = evenSolutionBufferOutput[i];
		oddSolutionBuffer[i] = oddSolutionBufferOutput[i];
		iterationLevel[i] = iterationLevelOutput[i];
	}
}

void swapBuffers(float * evenSolutionBuffer, float * oddSolutionBuffer, uint32_t * iterationLevel, float * evenSolutionBufferOutput, float * oddSolutionBufferOutput, uint32_t * iterationLevelOutput)
{
	float * tmp;
	uint32_t * tmpInt;
	tmp = evenSolutionBuffer; evenSolutionBuffer = evenSolutionBufferOutput; evenSolutionBufferOutput = tmp;
	tmp = oddSolutionBuffer; oddSolutionBuffer = oddSolutionBufferOutput; oddSolutionBufferOutput = tmp;
	tmpInt = iterationLevel; iterationLevel = iterationLevelOutput; iterationLevelOutput = tmpInt;
}

 
