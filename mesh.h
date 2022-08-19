void construct2DConnectivity(linearSystem &matrix, uint32_t N)
{
    matrix.indexPtr[0] = 0;
    int idx = 0;
    uint32_t dof, leftDOF, rightDOF, topDOF, bottomDOF;
    uint32_t idxPtrShift = 0;
    
    // Loop over the grid
    for (int iy = 0; iy < N; iy++) {
        for (int ix = 0; ix < N; ix++) {
            // Identify the center DOF
            dof = ix + iy * N;
            // Identify the neighbor DOFs
            leftDOF = dof - 1;
            rightDOF = dof + 1;
            topDOF = dof + N;
            bottomDOF = dof - N;
            // Reset index Pointer shift
            idxPtrShift = 1;
            // Left
            if (ix > 0) {
                matrix.nodeNeighbors[idx] = leftDOF;
                idx += 1;
                idxPtrShift += 1;
            }
            // Right
            if (ix < N-1) {
                matrix.nodeNeighbors[idx] = rightDOF;
                idx += 1;
                idxPtrShift += 1;
            }
            // Center
            matrix.nodeNeighbors[idx] = dof;
            // Bottom
            if (iy > 0) {
                matrix.nodeNeighbors[idx] = bottomDOF;
                idx += 1;
                idxPtrShift += 1;
            }
            // Top
            if (iy > N-1) {
                matrix.nodeNeighbors[idx] = topDOF;
                idx += 1;
                idxPtrShift += 1;
            }
            // Set the idxPtr entry values
            matrix.indexPtr[dof+1] = matrix.indexPtr[dof] + idxPtrShift;
        }
    }
}

void construct2DConnectivity_DiagonalLinks(linearSystem &matrix, uint32_t N)
{
	// Initialize neighbor DOF variables
    uint32_t swDOF, bottomDOF, seDOF, leftDOF, rightDOF, nwDOF, topDOF, neDOF, dof;

    // Initialize index Pointer array
    matrix.indexPtr[0] = 0;
    int idx = 0;
    uint32_t idxPtrShift = 0;

	// Compute dx
	float dx = 1.0 / (N+1);
    
    // Loop over the grid
    for (int iy = 0; iy < N; iy++) {
        for (int ix = 0; ix < N; ix++) {
            // Identify the center DOF
            dof = ix + iy * N;
            // Identify all 8 Cartesian Neighbor DOFs
			swDOF = (ix-1) + (iy-1)*N;
			bottomDOF = (ix) + (iy-1)*N;
			seDOF = (ix+1) + (iy-1)*N;
            leftDOF = (ix-1) + iy*N;
            rightDOF = (ix+1) + iy*N;
			nwDOF = (ix-1) + (iy+1) * N;
            topDOF = ix + (iy+1) * N;
            neDOF = (ix+1) + (iy+1) * N;
            // Reset index Pointer shift
            idxPtrShift = 0;
			// SW
			if ((ix > 0) && (iy > 0)) {
                matrix.nodeNeighbors[idx] = swDOF;
                matrix.offdiags[idx] = 0.0;
                idx += 1;
                idxPtrShift += 1;
			}
            // Bottom
            if (iy > 0) {
                matrix.nodeNeighbors[idx] = bottomDOF;
                matrix.offdiags[idx] = -1.0/(dx*dx);
                idx += 1;
                idxPtrShift += 1;
            }
            // SE
            if ((ix < N-1) && (iy > 0)) {
                matrix.nodeNeighbors[idx] = seDOF;
                matrix.offdiags[idx] = 0.0;
                idx += 1;
                idxPtrShift += 1;
            }
            // Left
            if (ix > 0) {
                matrix.nodeNeighbors[idx] = leftDOF;
                matrix.offdiags[idx] = -1.0/(dx*dx);
                idx += 1;
                idxPtrShift += 1;
            }
            // Right
            if (ix < N-1) {
                matrix.nodeNeighbors[idx] = rightDOF;
                matrix.offdiags[idx] = -1.0/(dx*dx);
                idx += 1;
                idxPtrShift += 1;
            }
			// NW
			if ((ix > 0) && (iy < N-1)) {
                matrix.nodeNeighbors[idx] = nwDOF;
                matrix.offdiags[idx] = 0.0;
                idx += 1;
                idxPtrShift += 1;
			}
            // Top
            if (iy < N-1) {
                matrix.nodeNeighbors[idx] = topDOF;
                matrix.offdiags[idx] = -1.0/(dx*dx);
                idx += 1;
                idxPtrShift += 1;
            }
            // NE
            if ((ix < N-1) && (iy < N-1)) {
                matrix.nodeNeighbors[idx] = neDOF;
                matrix.offdiags[idx] = 0.0;
                idx += 1;
                idxPtrShift += 1;
            }
            // Set the diag Inv entry    
			matrix.diagInv[dof] = (dx * dx) / 4.0;
            // Set the idxPtr entry values
            matrix.indexPtr[dof+1] = matrix.indexPtr[dof] + idxPtrShift;
        }
    }
}
