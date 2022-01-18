using namespace std; 
#include "inttypes.h"
#include "global.h"
#include <vector>
#include <set>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "structs.h"
#include "helper.h"
#include "mesh.h"
#include "seeds.h"
#include "swept_functions.h"

/* TO DOs:
1 - Get this guy to work with larger subdomains (>6 by 6 problem where each thread may need to update multiple DOFs) - OKish this seems to work (I need to understand the number of expansion steps a bit better but it seems to work.
2 - Actually implement Jacobi (not just a plus 1 update to the solution array) 
3 - Other seed locations (maybe 9 seeds or 16 seeds). Eventually just go to arbitrary seeds.
3b - Implement a way to allocate the GPU memory within the data structure, we do this every time we enter advance. If this could be a variable associated with the struct that would be ideal to avoid this allocation.
*/

int main (int argc, char * argv[]) 
{
    // Inputs
    uint32_t numIterationsGlobal = 1;
    uint32_t N = 10;

    // Define solution
    // Create solution containers on the CPU
	uint32_t Ndofs = N*N;
    float *du0 = new float[Ndofs];
    float *du1 = new float[Ndofs];
    initializeToZeros(du0, Ndofs);
    initializeToZeros(du1, Ndofs);
    // Create solution containers on the GPU
    float *du0_d;
    float *du1_d;
    cudaMalloc(&du0_d, sizeof(float) * Ndofs);
    cudaMalloc(&du1_d, sizeof(float) * Ndofs);
    cudaMemcpy(du0_d, du0, sizeof(float) * Ndofs, cudaMemcpyHostToDevice);
    cudaMemcpy(du1_d, du1, sizeof(float) * Ndofs, cudaMemcpyHostToDevice);
 
    // Define rhs on the CPU and GPU
    float *rhs = new float[Ndofs];
    initializeToOnes(rhs, Ndofs);
    float *rhs_d;
    cudaMalloc(&rhs_d, sizeof(float) * Ndofs);
    cudaMemcpy(rhs_d, rhs, sizeof(float) * Ndofs, cudaMemcpyHostToDevice);
	printHostSolutionFloat(rhs, 36, 6);
    // Perform iterations in global memory
    // globalMemorySolve2D(du1_d, du0_d, N, numIterations); 

	/* SHARED MEMORY ALGORITHM */ 

	// Create matrix object
	matrixInfo matrix;
	matrix.Ndofs = N*N;
	matrix.numEntries = (N-1)*(N-1)*8 + 4*(N-2)*5 + 4*3;

	// Create the idxPtr and nodeNeighbors for 2D Structured
	// Allocate matrix data on the CPU
	initializeMatrixHost(matrix);
    construct2DConnectivity_DiagonalLinks(matrix, N);

    // Allocate matrix data structures to the GPU 
    allocateMatrixDevice(matrix);
    copyMatrixDevice(matrix);

	/*************** SHARED MEMORY START **************************/
	
	// Initialize iteration level
	uint32_t * iterationLevel, * iterationLevel_d;
	iterationLevel = new uint32_t[Ndofs];
	initializeToZerosInt(iterationLevel, Ndofs);
	
	/*************** UPPER PYRAMIDAL STAGE ************************/

	printf("==================== PERFORMING UPPER PYRAMIDAL PARTITIONING =========================\n");
	
	// Create the partition for this stage
	meshPartitionForStage upperPyramidal;

    // Create initial seeds vector
    upperPyramidal.numSubdomains = 4;
	constructSeedsUpperPyramidal(upperPyramidal, N);
	for (auto setOfDofs : upperPyramidal.seeds) {
		for (auto dof : setOfDofs) {
			// printf("dof = %d\n", dof);
		}
	}

	// Create territories
	uint32_t numExpansionSteps = N/2-2;
	upperPyramidal.distanceFromSeed = new uint32_t[Ndofs];
	seedsExpandIntoSubdomains(upperPyramidal, matrix, iterationLevel, numExpansionSteps);
	expandToHaloRegions(upperPyramidal, matrix);
	printf("Distance\n");
	printHostSolutionInt(upperPyramidal.distanceFromSeed, Ndofs, N);

	// Create territory data and territory Index Ptr
	// Allocate memory on the GPU for all non-matrix related data
	createTerritoriesHost(upperPyramidal);
	// allocateTerritoriesDevice(upperPyramidal, Ndofs);
	// copyTerritoriesDevice(upperPyramidal, Ndofs);

	// Iteration Level
    cudaMalloc(&iterationLevel_d, sizeof(uint32_t) * Ndofs);
	cudaMemcpy(iterationLevel_d, iterationLevel, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);

	// Call Advance Kernel in shared memory
	numExpansionSteps = N/2-1;
	// Standard +1 Function
	advanceFunction(du0_d, iterationLevel_d, upperPyramidal, matrix, numExpansionSteps);
	// Actual Jacobi Iteration
	// advanceFunctionJacobi(du0_d, iterationLevel_d, upperPyramidal, matrix, numExpansionSteps, rhs_d, N);
	printf("Solution\n");
	printDeviceSolutionFloat(du0_d, Ndofs, N);
	printf("Iteration Level\n");
	printDeviceSolutionInt(iterationLevel_d, Ndofs, N);
	
	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);

	/*************** BRIDGE STAGE ************************/
	
	printf("==================== PERFORMING BRIDGE PARTITIONING =========================\n");
	
	// Create the partition for this stage
	meshPartitionForStage bridge;

    // Create initial seeds vector
    bridge.numSubdomains = 4;
	constructSeedsBridge(bridge, N);
	for (auto setOfDofs : bridge.seeds) {
		for (auto dof : setOfDofs) {
			// printf("dof = %d\n", dof);
		}
	}

	// Create territories
	numExpansionSteps = N/2-2;
	bridge.distanceFromSeed = new uint32_t[Ndofs];
	seedsExpandIntoSubdomains(bridge, matrix, iterationLevel, numExpansionSteps);
	expandToHaloRegions(bridge, matrix);
	printf("Distance\n");
	printHostSolutionInt(bridge.distanceFromSeed, Ndofs, N);

	// Create territory data and territory Index Ptr
	// Allocate memory on the GPU for all non-matrix related data
	createTerritoriesHost(bridge);
	// allocateTerritoriesDevice(upperPyramidal, Ndofs);
	// copyTerritoriesDevice(upperPyramidal, Ndofs);

	// Call Advance Kernel in shared memory
	numExpansionSteps = N/2-1;
	advanceFunction(du0_d, iterationLevel_d, bridge, matrix, numExpansionSteps);
	printf("Solution\n");
	printDeviceSolutionFloat(du0_d, Ndofs, N);
	printf("Iteration Level\n");
	printDeviceSolutionInt(iterationLevel_d, Ndofs, N);
	
	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);
	
	/*************** LOWER PYRAMIDAL STAGE ************************/

	printf("==================== PERFORMING LOWER PYRAMIDAL PARTITIONING =========================\n");
	
	// Create the partition for this stage
	meshPartitionForStage lowerPyramidal;

    // Create initial seeds vector
    lowerPyramidal.numSubdomains = 1;
	constructSeedsLowerPyramidal(lowerPyramidal, N);
	for (auto setOfDofs : lowerPyramidal.seeds) {
		for (auto dof : setOfDofs) {
			// printf("dof = %d\n", dof);
		}
	}

	// Create territories
	numExpansionSteps = N-4; // N/2-1;
	lowerPyramidal.distanceFromSeed = new uint32_t[Ndofs];
	seedsExpandIntoSubdomains(lowerPyramidal, matrix, iterationLevel, numExpansionSteps);
	expandToHaloRegions(lowerPyramidal, matrix);
	for (auto elem : lowerPyramidal.territoriesExpanded[0]) {
		// printf("Elem %d\n", elem);
	}
	printf("Distance\n");
	printHostSolutionInt(lowerPyramidal.distanceFromSeed, Ndofs, N);

	// Create territory data and territory Index Ptr
	// Allocate memory on the GPU for all non-matrix related data
	createTerritoriesHost(lowerPyramidal);
	// allocateTerritoriesDevice(upperPyramidal, Ndofs);
	// copyTerritoriesDevice(upperPyramidal, Ndofs);

	// Call Advance Kernel in shared memory
	numExpansionSteps = N-2; // 4;
	advanceFunction(du0_d, iterationLevel_d, lowerPyramidal, matrix, numExpansionSteps);

	printf("Solution\n");
	printDeviceSolutionFloat(du0_d, Ndofs, N);
	printf("Iteration Level\n");
	printDeviceSolutionInt(iterationLevel_d, Ndofs, N);
	
	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);

	/*************** DUAL BRIDGE STAGE ************************/
	
	printf("==================== PERFORMING DUAL BRIDGE PARTITIONING =========================\n");

	// Create the partition for this stage
	meshPartitionForStage dualBridge;

    // Create initial seeds vector
    dualBridge.numSubdomains = 4;
	constructSeedsDualBridge(dualBridge, N);
	for (auto setOfDofs : dualBridge.seeds) {
		for (auto dof : setOfDofs) {
			// printf("dof = %d\n", dof);
		}
	}

	// Create territories
	numExpansionSteps = N-3; // N/2; // N/2;
	dualBridge.distanceFromSeed = new uint32_t[Ndofs];
	seedsExpandIntoSubdomains(dualBridge, matrix, iterationLevel, numExpansionSteps);
	expandToHaloRegions(dualBridge, matrix);
	printf("Distance\n");
	printHostSolutionInt(dualBridge.distanceFromSeed, Ndofs, N);

	// Create territory data and territory Index Ptr
	// Allocate memory on the GPU for all non-matrix related data
	createTerritoriesHost(dualBridge);
	// allocateTerritoriesDevice(upperPyramidal, Ndofs);
	// copyTerritoriesDevice(upperPyramidal, Ndofs);
	for (int i = 0; i < matrix.Ndofs; i++) {
		// printf("seed[%d] = %d\n", i, lowerPyramidal.distanceFromSeed[i]); 
	}

	// Call Advance Kernel in shared memory
	numExpansionSteps = N-2;
	advanceFunction(du0_d, iterationLevel_d, dualBridge, matrix, numExpansionSteps);
	printf("Solution\n");
	printDeviceSolutionFloat(du0_d, Ndofs, N);
	printf("Iteration Level\n");
	printDeviceSolutionInt(iterationLevel_d, Ndofs, N);
	
	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);

	/* Repeat all steps together */
/*
    initializeToZeros(du0, matrix.Ndofs);
    initializeToZerosInt(iterationLevel, matrix.Ndofs);
    cudaMemcpy(du0_d, du0, sizeof(float) * Ndofs, cudaMemcpyHostToDevice);
    cudaMemcpy(iterationLevel_d, iterationLevel, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
	// printDeviceSolutionFloat(du0_d, Ndofs, N);
	numExpansionSteps = 0;
		
	printf("======================= ALGORITHM START =========================================\n");
	
	for (int sweptIteration = 0; sweptIteration < 0; sweptIteration++) {

		printf("======================= CYCLE %d =========================================\n", sweptIteration);
		
		// Stage 1 - Upper Pyramidal
		numExpansionSteps += (N/2-1);
		printf("UPPER PYRAMIDAL STAGE\n");
		advanceFunction(du0_d, iterationLevel_d, upperPyramidal, matrix, numExpansionSteps);
		printDeviceSolutionFloat(du0_d, Ndofs, N);
		
		// Stage 2 - Bridge Stage
		printf("BRIDGE STAGE\n");
		advanceFunction(du0_d, iterationLevel_d, bridge, matrix, numExpansionSteps);
		printDeviceSolutionFloat(du0_d, Ndofs, N);

		// Lower Pyramidal
		numExpansionSteps += (N/2-1);
		printf("LOWER PYRAMIDAL STAGE\n");
		advanceFunction(du0_d, iterationLevel_d, lowerPyramidal, matrix, numExpansionSteps);
		printDeviceSolutionFloat(du0_d, Ndofs, N);

		// Dual Bridge
		printf("DUAL BRIDGE STAGE\n");
		advanceFunction(du0_d, iterationLevel_d, dualBridge, matrix, numExpansionSteps);
		printDeviceSolutionFloat(du0_d, Ndofs, N);

	}	

*/
}
