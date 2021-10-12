void sweptIteration(float * du1, float * du0)
{
    // Upper Pyramidal 
    vector<subdomains> upSubs;
    findSeedsAndSubdomains();
    advance()

    // Bridge
    vector<subdomains> bSubs;
    findSeedsAndSubdomains();
    advance();

    // Lower Pyramidal 
    vector<subdomains> lpSubs;
    findSeedsAndSubdomains();
    advance()

    // Dual Bridge()
    vector<subdomains> dbSubdomains;
    findSeedsAndSubdomains();
    advance()
}

__global__
void advance(vector<subdomain> listOfSubdomains)
{
    // The ith block works with the ith listOfSubdomains
    
    
}

void sharedMemorySolve2D(float * du1, float * du0, uint32_t N, uint32_t numIters)
{
    // Define the number of threads per block in each direction
    uint32_t tpbx = 32;
    uint32_t tpby = 32;
    dim3 threadsPerBlock(tpbx, tpby);

    // Define the number of blocks in each direction
    uint32_t nBlocksx = ceil((float)N / tpbx);
    uint32_t nBlocksy = ceil((float)N / tpbx);
    dim3 blocks(nBlocksx, nBlocksy);

    // Create a temporary variable for the solution swap
    float * tmp;
    
    // Perform all Jacobi iterations
    for (int sweptIter = 0; sweptIter < numSweptIters; sweptIter++) {
        // Perform stencil update on global memory
        sweptIteration(du1, du0, N);
    }
}

__global__
void upperPyramidalStep(vector<set<uint32_t>> vectorOfPartitionedDOFS)
{
	// Move the solution from global memory to shared memory

    // Apply the update within shared memory

    // Move solution from shared to global memory

}
