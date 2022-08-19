using namespace std; 
#include "inttypes.h"
#include <vector>
#include <set>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "structs.h"
#include "helper.h"
#include "mesh.h"
#include "seeds_largeSubdomains.h"
#include "subdomains_largeSubdomains_v2.h"
#include "toDevice.h"
#include "global.h"
#include "swept_update_jacobi_testing_v2.h"
#include "residual.h"
#include "initialize.h"
#include "matrix.h"
#include "print.h"
#include "debug.h"

#include <chrono>

# define PRINT_SOLUTION

/* TO DOs:
1 - Get this guy to work with larger subdomains (>6 by 6 problem where each thread may need to update multiple DOFs) - OKish this seems to work (I need to understand the number of expansion steps a bit better but it seems to work.
2 - Actually implement Jacobi (not just a plus 1 update to the solution array) 
3 - Other seed locations (maybe 9 seeds or 16 seeds). Eventually just go to arbitrary seeds.
3b - Implement a way to allocate the GPU memory within the data structure, we do this every time we enter advance. If this could be a variable associated with the struct that would be ideal to avoid this allocation.
*/

int main (int argc, char * argv[]) 
{
    // Inputs (sometimes this works...maybe need some syncs)
    uint32_t N = 60;
    uint32_t nPerSub = 6; // each subdomain is Nsub by Nsub
	uint32_t numSweptCycles = 1;
	// Number of subdomains in 1D direction
	uint32_t nSub = N / nPerSub;
	
	// Define the linear system Ax = b

	// Initialize the linear system and allocate matrix data structures (indexPtr, nodeNeighbors, offdiags) on the host
	linearSystem matrix;
	fillMatrixNumDOFsEntriesDiagonalLinks(matrix, N);
	initializeMatrixHost(matrix, N);
    construct2DConnectivity_DiagonalLinks(matrix, N); // 2D structured domain with diagonal links
	constructRhs(matrix);
    
	// Allocate matrix data structures to the GPU (is this even necessary? - yes! used by global memory solution) 
	linearSystemDevice matrix_d;
    allocateMatrixDevice(matrix_d, matrix);
    copyMatrixDevice(matrix_d, matrix);

	// SHARED MEMORY START 
	
	printf("==================== SHARED MEMORY ALGORITHM =========================\n");
	
	// Initialize iteration level
	uint32_t Ndofs = matrix.Ndofs;
	uint32_t * iterationLevel, * iterationLevel_d;
	iterationLevel = new uint32_t[Ndofs];
	initializeToZerosInt(iterationLevel, Ndofs);
	// Iteration Level
    cudaMalloc(&iterationLevel_d, sizeof(uint32_t) * Ndofs);
	cudaMemcpy(iterationLevel_d, iterationLevel, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
	
	// Create swept solution variables on the device
	float * bottomSweptSolutionIn_d, * topSweptSolutionIn_d;
	cudaMalloc(&topSweptSolutionIn_d, sizeof(float) * matrix.Ndofs);
	cudaMalloc(&bottomSweptSolutionIn_d, sizeof(float) * matrix.Ndofs);
	uint32_t threadsPerBlock = 256;
	uint32_t numBlocks = ceil((float)matrix.Ndofs / threadsPerBlock);
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(topSweptSolutionIn_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(bottomSweptSolutionIn_d, matrix.Ndofs);	
	float * bottomSweptSolutionOut_d, * topSweptSolutionOut_d;
	cudaMalloc(&topSweptSolutionOut_d, sizeof(float) * matrix.Ndofs);
	cudaMalloc(&bottomSweptSolutionOut_d, sizeof(float) * matrix.Ndofs);

	// Number of expansion steps
	uint32_t numExpansionSteps = nPerSub-1; 

	// Number of Jacobi steps
	uint32_t numJacobiSteps = 0;
	uint32_t numJacobiStepsIncrement = (nPerSub-1)/2+1;
	
	/*************** STAGE 1: UPPER PYRAMIDAL STAGE ************************/
	
	printf("==================== PERFORMING UPPER PYRAMIDAL PARTITIONING =========================\n");
	
	// HOST
	// Create the partition for this stage (constructing seeds, territories, and local matrix data structures)
	meshPartitionForStage upperPyramidal;
	constructSeedsHost(upperPyramidal, N, nSub, 0);
	constructTerritoriesHost(upperPyramidal, matrix, iterationLevel, N, nSub, 0);
	constructLocalMatricesHost(upperPyramidal, matrix);

	// DEVICE
	// Allocate and copy all partition related data to the GPU
	meshPartitionForStageDevice upperPyramidal_d;
	allocatePartitionDevice(upperPyramidal_d, upperPyramidal, Ndofs);
	copyPartitionDevice(upperPyramidal_d, upperPyramidal, Ndofs);

	// JACOBI
	// Perform Jacobi Iteration kernel call
	determineSharedMemoryAllocation(upperPyramidal);
	numJacobiSteps += numJacobiStepsIncrement;
	stageAdvanceJacobiPerformance<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(topSweptSolutionOut_d, bottomSweptSolutionOut_d, topSweptSolutionIn_d, bottomSweptSolutionIn_d, iterationLevel_d, upperPyramidal_d, 0, numJacobiSteps, numJacobiSteps);

	// POSTPROCESSING	
	// Print solution
	printPartitionInformation(upperPyramidal, matrix, topSweptSolutionOut_d, iterationLevel_d, N);	
	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	
	// Compute L2 residual
	float residual = computeL2Residual(topSweptSolutionOut_d, matrix_d);
	printf("The residual is %f\n", residual);
	// Swap vectors
	float * tmp;
	tmp = topSweptSolutionIn_d; topSweptSolutionIn_d = topSweptSolutionOut_d; topSweptSolutionOut_d = topSweptSolutionIn_d;
	tmp = bottomSweptSolutionIn_d; bottomSweptSolutionIn_d = bottomSweptSolutionOut_d; bottomSweptSolutionOut_d = bottomSweptSolutionIn_d;
	
	/*************** STAGE 2: BRIDGE STAGE ************************/

	printf("==================== PERFORMING BRIDGE PARTITIONING =========================\n");
	
	// HOST
	// Create the partition for this stage (constructing seeds, territories, and local matrix data structures)
	meshPartitionForStage bridge;
	constructSeedsHost(bridge, N, nSub, 1);
	constructTerritoriesHost(bridge, matrix, iterationLevel, N, nSub, 1);
	constructLocalMatricesHost(bridge, matrix);
	
	// DEVICE
	// Allocate and copy all partition related data to the GPU
	meshPartitionForStageDevice bridge_d;
	allocatePartitionDevice(bridge_d, bridge, Ndofs);
	copyPartitionDevice(bridge_d, bridge, Ndofs);
	
	// JACOBI
	// Perform Jacobi Iteration kernel call
	determineSharedMemoryAllocation(bridge);
	stageAdvanceJacobiPerformance<<<bridge.numSubdomains, 512, bridge.sharedMemorySize>>>(topSweptSolutionOut_d, bottomSweptSolutionOut_d, topSweptSolutionIn_d, bottomSweptSolutionIn_d, iterationLevel_d, bridge_d, 0, numJacobiSteps, numJacobiSteps);
	
	// POSTPROCESSING	
	// Print solution
	printPartitionInformation(bridge, matrix, topSweptSolutionOut_d, iterationLevel_d, N);	
	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	
	// Compute L2 residual
	residual = computeL2Residual(topSweptSolutionOut_d, matrix_d);
	printf("The residual is %f\n", residual);
	// Swap vectors
	tmp = topSweptSolutionIn_d; topSweptSolutionIn_d = topSweptSolutionOut_d; topSweptSolutionOut_d = topSweptSolutionIn_d;
	tmp = bottomSweptSolutionIn_d; bottomSweptSolutionIn_d = bottomSweptSolutionOut_d; bottomSweptSolutionOut_d = bottomSweptSolutionIn_d;
	
	/*************** STAGE 3: LOWER PYRAMIDAL STAGE ************************/
/*
	printf("==================== PERFORMING LOWER PYRAMIDAL PARTITIONING =========================\n");

	// HOST
	// Create the partition for this stage (constructing seeds, territories, and local matrix data structures)
	meshPartitionForStage lowerPyramidal;
	constructSeedsHost(lowerPyramidal, N, nSub, 2);
	constructTerritoriesHost(lowerPyramidal, matrix, iterationLevel, N, nSub, 2);
	constructLocalMatricesHost(lowerPyramidal, matrix);
	
	// DEVICE
	// Allocate and copy all partition related data to the GPU
	meshPartitionForStageDevice lowerPyramidal_d;
	allocatePartitionDevice(lowerPyramidal_d, lowerPyramidal, Ndofs);
	copyPartitionDevice(lowerPyramidal_d, lowerPyramidal, Ndofs);
	
	// JACOBI
	// Perform Jacobi Iteration kernel call
	determineSharedMemoryAllocation(lowerPyramidal, false);
	numJacobiSteps += numJacobiStepsIncrement;
	stageAdvanceJacobiPerformance<<<lowerPyramidal.numSubdomains, 512, lowerPyramidal.sharedMemorySize>>>(topSweptSolutionOut_d, bottomSweptSolutionOut_d, topSweptSolutionIn_d, bottomSweptSolutionIn_d, iterationLevel_d, lowerPyramidal_d, 0, numJacobiSteps, numJacobiSteps);

	// POSTPROCESSING	
	// Print solution
	printPartitionInformation(lowerPyramidal, matrix, topSweptSolutionOut_d, iterationLevel_d, N);	
	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	
	// Compute L2 residual
	residual = computeL2Residual(topSweptSolutionOut_d, matrix_d);
	printf("The residual is %f\n", residual);
	// Swap vectors
	tmp = topSweptSolutionIn_d; topSweptSolutionIn_d = topSweptSolutionOut_d; topSweptSolutionOut_d = topSweptSolutionIn_d;
	tmp = bottomSweptSolutionIn_d; bottomSweptSolutionIn_d = bottomSweptSolutionOut_d; bottomSweptSolutionOut_d = bottomSweptSolutionIn_d;
*/	
	/*************** STAGE 4: DUAL BRIDGE STAGE ************************/
/*	
	printf("==================== PERFORMING DUAL BRIDGE PARTITIONING =========================\n");

	// HOST
	// Create the partition for this stage (constructing seeds, territories, and local matrix data structures)
	meshPartitionForStage dualBridge;
	constructSeedsHost(dualBridge, N, nSub, 3);
	constructTerritoriesHost(dualBridge, matrix, iterationLevel, N, nSub, 3);
	constructLocalMatricesHost(dualBridge, matrix);
	
	// DEVICE
	// Allocate and copy all partition related data to the GPU
	meshPartitionForStageDevice dualBridge_d;
	allocatePartitionDevice(dualBridge_d, dualBridge, Ndofs);
	copyPartitionDevice(dualBridge_d, dualBridge, Ndofs);
	
	// JACOBI
	// Perform Jacobi Iteration kernel call
	determineSharedMemoryAllocation(dualBridge);
	stageAdvanceJacobiPerformance<<<dualBridge.numSubdomains, 512, dualBridge.sharedMemorySize>>>(topSweptSolutionOut_d, bottomSweptSolutionOut_d, topSweptSolutionIn_d, bottomSweptSolutionIn_d, iterationLevel_d, dualBridge_d, numJacobiStepsIncrement, numJacobiSteps, numJacobiSteps);

	// POSTPROCESSING	
	// Print solution
	printPartitionInformation(dualBridge, matrix, topSweptSolutionOut_d, iterationLevel_d, N);	
	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	
	// Compute L2 residual
	residual = computeL2Residual(topSweptSolutionOut_d, matrix_d);
	printf("The residual is %f\n", residual);
	// Swap vectors
	tmp = topSweptSolutionIn_d; topSweptSolutionIn_d = topSweptSolutionOut_d; topSweptSolutionOut_d = topSweptSolutionIn_d;
	tmp = bottomSweptSolutionIn_d; bottomSweptSolutionIn_d = bottomSweptSolutionOut_d; bottomSweptSolutionOut_d = bottomSweptSolutionIn_d;
*/	
	/*************** GLOBAL MEMORY START **************************/
	
	printf("==================== GLOBAL MEMORY ALGORITHM =========================\n");
    
    // Create solution containers on the CPU
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

	// Initial L2 residual
	// float residual, residualInit;
	float residualInit;
	residual = computeL2Residual(du0_d, matrix_d);
	residualInit = residual;
	printf("The initial residual is %f\n", residual);
	
	// Initialize solution and residual norm variables
	float * solutionGM = new float[matrix.Ndofs];
	float residualGM;

	// Number of total Jacobi iterations to perform
	uint32_t numIterations = 2; // ((nPerSub-1)/2+1)*2 * numSweptCycles; // 2 * (nPerSub-1) * numSweptCycles;
	//uint32_t numJacobiStepsIncrement2 = (((nPerSub-1)/2)*2+1) - ((nPerSub-1)/2 + 1); // nPerSub - ((nPerSub-1)/2 + 1);

	// Perform global memory iterations
	float globalTime;	
	cudaEvent_t start_g, stop_g;
	cudaEventCreate(&start_g);
	cudaEventCreate(&stop_g);
	cudaEventRecord(start_g, 0);
    globalMemorySolve2D(du1_d, du0_d, matrix_d, N, numIterations);
	cudaEventRecord(stop_g, 0);
	cudaEventSynchronize(stop_g);
	cudaEventElapsedTime(&globalTime, start_g, stop_g);

	// Print solution
	if (numIterations % 2 == 0) { 
		// printDeviceSolutionFloat(du0_d, Ndofs, N);
		solutionGM = du0_d;
	}
	else if (numIterations % 2 == 1) { 
		// printDeviceSolutionFloat(du1_d, Ndofs, N);
		solutionGM = du1_d;
	}
	printf("Number of Iterations = %d\n", numIterations);
	
	// Initial L2 residual
	if (numIterations % 2 == 0) { 
		residual = computeL2Residual(du0_d, matrix_d);
	}
	else if (numIterations % 2 == 1) { 
		residual = computeL2Residual(du1_d, matrix_d);
	}
	printf("GLOBAL MEMORY ITERATIONS: The residual is %f\n", residual);
	residualGM = residual;
	printDeviceSimilarity(topSweptSolutionOut_d, solutionGM, 1e-12, Ndofs, N);

	/*************** SUBDOMAIN CONSTRUCTION COMPLETE - PERFORM ACTUAL ITERATIONS ON GPU ************************/
/*
	// Repeat all steps together 
	threadsPerBlock = 128;
	numBlocks = ceil((float)matrix.Ndofs / threadsPerBlock);
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(topSweptSolutionIn_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(bottomSweptSolutionIn_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(topSweptSolutionOut_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(bottomSweptSolutionOut_d, matrix.Ndofs);	
    initializeToZerosInt(iterationLevel, matrix.Ndofs);
    cudaMemcpy(iterationLevel_d, iterationLevel, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
	numJacobiSteps = 0;

    // initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(topSweptSolution_d, matrix.Ndofs);
    // initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(bottomSweptSolution_d, matrix.Ndofs);
    // initializeToZerosDeviceInt<<<numBlocks, threadsPerBlock>>>(iterationLevel_d, matrix.Ndofs);
	printf("INITIAL SOLUTION\n");
	// printDeviceSolutionFloat(topSweptSolution_d, Ndofs, N);
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errAsync != cudaSuccess) {
		printf("Setup kernel error: %s\n", cudaGetErrorString(errAsync));
	}
	else {
		printf("Setup Successful\n");
	}

	numJacobiSteps = 0;
	uint32_t numJacobiStepsMin = 0;

	printf("======================= CYCLE START =========================================\n");

	float time_total_1 = 0;
	float time_total_2 = 0;
	float time_total_3 = 0;
	float time_total_4 = 0;
	float time_total_5 = 0;	
	float time_stage_1, time_stage_2, time_stage_3, time_stage_4, time_stage_5;	
	cudaEvent_t start_1, stop_1, start_2, stop_2, start_3, stop_3, start_4, stop_4, start_5, stop_5;
	cudaEventCreate(&start_1);
	cudaEventCreate(&stop_1);
	cudaEventCreate(&start_2);
	cudaEventCreate(&stop_2);
	cudaEventCreate(&start_3);
	cudaEventCreate(&stop_3);
	cudaEventCreate(&start_4);
	cudaEventCreate(&stop_4);
	cudaEventCreate(&start_5);
	cudaEventCreate(&stop_5);

	for (int sweptIteration = 0; sweptIteration < numSweptCycles; sweptIteration++) {

		if (sweptIteration % 10 == 0) {
			printf("======================= CYCLE %d =========================================\n", sweptIteration);
		}

		// Set number of Jacobi iterations for first two stages
		numJacobiSteps += numJacobiStepsIncrement;

		// Stage 1 - Upper Pyramidal
		// printf("UPPER PYRAMIDAL STAGE\n");
		cudaEventRecord(start_1, 0);
		stageAdvanceJacobiPerformance<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(topSweptSolutionOut_d, bottomSweptSolutionOut_d, topSweptSolutionIn_d, bottomSweptSolutionIn_d, iterationLevel_d, upperPyramidal_d, numJacobiStepsMin, numJacobiSteps, numJacobiSteps);
		// cudaDeviceSynchronize();
		cudaEventRecord(stop_1, 0);
		cudaEventSynchronize(stop_1);
		cudaEventElapsedTime(&time_stage_1, start_1, stop_1);
		time_total_1 += time_stage_1;
		// printPartitionInformation(upperPyramidal, matrix, topSweptSolution_d, iterationLevel_d, N);	
		// Swap vectors
		tmp = topSweptSolutionIn_d; topSweptSolutionIn_d = topSweptSolutionOut_d; topSweptSolutionOut_d = topSweptSolutionIn_d;
		tmp = bottomSweptSolutionIn_d; bottomSweptSolutionIn_d = bottomSweptSolutionOut_d; bottomSweptSolutionOut_d = bottomSweptSolutionIn_d;
		
		// Set lower bound for Jacobi iterations
		if (sweptIteration > 0) {
			numJacobiStepsMin += numJacobiStepsIncrement-1;
		}

		// Stage 2 - Bridge Stage
		// printf("BRIDGE STAGE\n");
		cudaEventRecord(start_2, 0);
		stageAdvanceJacobiPerformance<<<bridge.numSubdomains, 512, bridge.sharedMemorySize>>>(topSweptSolutionOut_d, bottomSweptSolutionOut_d, topSweptSolutionIn_d, bottomSweptSolutionIn_d, iterationLevel_d, bridge_d, numJacobiStepsMin, numJacobiSteps, numJacobiSteps);
		// cudaDeviceSynchronize();
		cudaEventRecord(stop_2, 0);
		cudaEventSynchronize(stop_2);
		cudaEventElapsedTime(&time_stage_2, start_2, stop_2);
		time_total_2 += time_stage_2;
		// printPartitionInformation(upperPyramidal, matrix, topSweptSolution_d, iterationLevel_d, N);	
		// Swap vectors
		tmp = topSweptSolutionIn_d; topSweptSolutionIn_d = topSweptSolutionOut_d; topSweptSolutionOut_d = topSweptSolutionIn_d;
		tmp = bottomSweptSolutionIn_d; bottomSweptSolutionIn_d = bottomSweptSolutionOut_d; bottomSweptSolutionOut_d = bottomSweptSolutionIn_d;

		// Set number of Jacobi iterations for second two stages
		numJacobiSteps += numJacobiStepsIncrement;
		
		// Lower Pyramidal
		// printf("LOWER PYRAMIDAL STAGE\n");
		cudaEventRecord(start_3, 0);
		stageAdvanceJacobiPerformance<<<lowerPyramidal.numSubdomains, 512, lowerPyramidal.sharedMemorySize>>>(topSweptSolutionOut_d, bottomSweptSolutionOut_d, topSweptSolutionIn_d, bottomSweptSolutionIn_d, iterationLevel_d, lowerPyramidal_d, numJacobiStepsMin, numJacobiSteps, numJacobiSteps);
		// cudaDeviceSynchronize();
		cudaEventRecord(stop_3, 0);
		cudaEventSynchronize(stop_3);
		cudaEventElapsedTime(&time_stage_3, start_3, stop_3);
		time_total_3 += time_stage_3;
		// printPartitionInformation(upperPyramidal, matrix, topSweptSolution_d, iterationLevel_d, N);	
		// Swap vectors
		tmp = topSweptSolutionIn_d; topSweptSolutionIn_d = topSweptSolutionOut_d; topSweptSolutionOut_d = topSweptSolutionIn_d;
		tmp = bottomSweptSolutionIn_d; bottomSweptSolutionIn_d = bottomSweptSolutionOut_d; bottomSweptSolutionOut_d = bottomSweptSolutionIn_d;
		
		// Set lower bound for Jacobi iterations
		numJacobiStepsMin += numJacobiStepsIncrement-1;

		// Dual Bridge
		// printf("DUAL BRIDGE STAGE\n");
		cudaEventRecord(start_4, 0);
		stageAdvanceJacobiPerformance<<<dualBridge.numSubdomains, 512, dualBridge.sharedMemorySize>>>(topSweptSolutionOut_d, bottomSweptSolutionOut_d, topSweptSolutionIn_d, bottomSweptSolutionIn_d, iterationLevel_d, dualBridge_d, numJacobiStepsMin, numJacobiSteps, numJacobiSteps);
		// cudaDeviceSynchronize();
		cudaEventRecord(stop_4, 0);
		cudaEventSynchronize(stop_4);
		cudaEventElapsedTime(&time_stage_4, start_4, stop_4);
		time_total_4 += time_stage_4;
		// printPartitionInformation(upperPyramidal, matrix, topSweptSolution_d, iterationLevel_d, N);	
		// Swap vectors
		tmp = topSweptSolutionIn_d; topSweptSolutionIn_d = topSweptSolutionOut_d; topSweptSolutionOut_d = topSweptSolutionIn_d;
		tmp = bottomSweptSolutionIn_d; bottomSweptSolutionIn_d = bottomSweptSolutionOut_d; bottomSweptSolutionOut_d = bottomSweptSolutionIn_d;

	}

	// Set number of Jacobi iterations for final fill-in stage
	numJacobiSteps += numJacobiStepsIncrement;
	// Final Stage	
	printf("FINAL STAGE\n");
	// JACOBI
	cudaEventRecord(start_5, 0);
	// stageAdvanceJacobiPerformance<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(topSweptSolution_d, bottomSweptSolution_d, iterationLevel_d, upperPyramidal_d, 2*(nPerSub-1), 3*(nPerSub-1));
	stageAdvanceJacobiPerformance<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(topSweptSolutionOut_d, bottomSweptSolutionOut_d, topSweptSolutionIn_d, bottomSweptSolutionIn_d, iterationLevel_d, upperPyramidal_d, numJacobiStepsMin, numJacobiSteps - numJacobiStepsIncrement, numJacobiSteps);
	// cudaDeviceSynchronize();
	cudaEventRecord(stop_5, 0);
	cudaEventSynchronize(stop_5);
	cudaEventElapsedTime(&time_total_5, start_5, stop_5);

	// Print information
	
	printf("==================== FINAL INFORMATION =========================\n");
	
	// Print solution for swept algorithm

	printf("SOLUTION\n");
	printf("GLOBAL MEMORY SOLUTION AFTER %d ITERATIONS\n", numIterations);
	// printDeviceSolutionFloat(solutionGM, Ndofs, N);
	printf("SHARED MEMORY (SWEPT) SOLUTION AFTER %d ITERATIONS\n", numIterations);

	// Compute L2 residual
	residual = computeL2Residual(topSweptSolutionOut_d, matrix_d);
	printf("RESIDUAL\n");
	printf("Global: The initial residual was %f while the final residual is %f\n", residualInit, residualGM);
	printf("Swept: The initial residual was %f while the final residual is %f\n", residualInit, residual);

	// Print times for global and shared memory to complete
	float sweptTime = time_total_1 + time_total_2 + time_total_3 + time_total_4 + time_total_5;
	printf("TIME INFORMATION\n");
	printf("Global: Elapsed time in ms %f\n", globalTime);
	printf("Swept: Elapsed time in microseconds %f\n", sweptTime);
	printf("Time for upper pyramidal is %f\n", time_total_1);
	printf("Time for bridge is %f\n", time_total_2);
	printf("Time for lower pyramidal is %f\n", time_total_3);
	printf("Time for dual bridge is %f\n", time_total_4);
	printf("Time for final step is %f\n", time_total_5);
	printf("Speedup is %f\n", globalTime / sweptTime);
*/
}

