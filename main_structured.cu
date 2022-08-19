using namespace std; 
#include "inttypes.h"
#include <algorithm>
#include <vector>
#include <set>
#include <map>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "structs.h"
#include "helper.h"
#include "mesh.h"
#include "seeds.h"
#include "subdomains.h"
#include "initialize.h"
#include "toDevice.h"
#include "global.h"
#include "swept_update_jacobi.h"
#include "residual.h"
#include "matrix.h"
#include "print.h"
#include "debug.h"

// # define PRINT_SOLUTION

int main (int argc, char * argv[]) 
{
    // INPUTS
    uint32_t N = 1024; //1024;
    uint32_t nPerSub = 16; // 16; // each subdomain is Nsub by Nsub
	uint32_t numSweptCycles = 10000; // 10000;
	uint32_t nSub = N / nPerSub; // Number of subdomains in 1D direction
	
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

	float * evenSolutionBuffer_d, * oddSolutionBuffer_d, * solution_d;
	cudaMalloc(&evenSolutionBuffer_d, sizeof(float) * matrix.Ndofs);
	cudaMalloc(&oddSolutionBuffer_d, sizeof(float) * matrix.Ndofs);
	cudaMalloc(&solution_d, sizeof(float) * matrix.Ndofs);
	uint32_t threadsPerBlock = 256;
	uint32_t numBlocks = ceil((float)matrix.Ndofs / threadsPerBlock);
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(evenSolutionBuffer_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(oddSolutionBuffer_d, matrix.Ndofs);	

	/*************** STAGE 1: UPPER PYRAMIDAL STAGE ************************/
	
	printf("==================== PERFORMING UPPER PYRAMIDAL PARTITIONING =========================\n");

	// INITIALIZE
	meshPartitionForStage upperPyramidal;
	uint32_t numExpansionSteps = (nPerSub-1)/2-1; 
	uint32_t numJacobiSteps = (nPerSub-1)/2;
	
	// HOST
	constructSeedsHost(upperPyramidal, N, nSub, 0);
	seedsExpandIntoSubdomains(upperPyramidal, matrix, iterationLevel, numExpansionSteps);
	createHaloRegions(upperPyramidal, matrix);
	determineIterationLevelPerDOF(upperPyramidal, matrix, iterationLevel, 0, numJacobiSteps);
	createTerritoriesHost(upperPyramidal);
	constructLocalMatricesHost(upperPyramidal, matrix);

	// DEVICE
	meshPartitionForStageDevice upperPyramidal_d;
	allocatePartitionDevice(upperPyramidal_d, upperPyramidal, Ndofs);
	copyPartitionDevice(upperPyramidal_d, upperPyramidal, Ndofs);
	
	// JACOBI
	determineSharedMemoryAllocation(upperPyramidal);
	stageAdvanceJacobiPerformance<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, upperPyramidal_d, 0, numJacobiSteps);
	
	// POSTPROCESSING
	assembleSolutionFromBuffers<<<numBlocks, threadsPerBlock>>>(solution_d, evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, Ndofs);
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	
#ifdef PRINT_SOLUTION
	// Print solution
	printHostSolutionInt(upperPyramidal.subdomainOfDOFs, matrix.Ndofs, N);	
	printDeviceSolutionInt(iterationLevel_d, matrix.Ndofs, N);	
#endif
	printf("The number of subdomains are %d\n", upperPyramidal.numSubdomains);
	
	/*************** STAGE 2: BRIDGE STAGE ************************/
	
	printf("==================== PERFORMING BRIDGE PARTITIONING =========================\n");
	
	// INITIALIZE
	meshPartitionForStage bridge;

	// HOST
	constructSeedsHost(bridge, N, nSub, 1);
	seedsExpandIntoSubdomains(bridge, matrix, iterationLevel, numExpansionSteps);
	createHaloRegions(bridge, matrix);
	determineIterationLevelPerDOF(bridge, matrix, iterationLevel, 0, numJacobiSteps);
	createTerritoriesHost(bridge);
	constructLocalMatricesHost(bridge, matrix);
	
	// DEVICE
	meshPartitionForStageDevice bridge_d;
	allocatePartitionDevice(bridge_d, bridge, Ndofs);
	copyPartitionDevice(bridge_d, bridge, Ndofs);
	
	// JACOBI
	determineSharedMemoryAllocation(bridge);
	stageAdvanceJacobiPerformance<<<bridge.numSubdomains, 512, bridge.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, bridge_d, 0, numJacobiSteps);
	
	// POSTPROCESSING	
	assembleSolutionFromBuffers<<<numBlocks, threadsPerBlock>>>(solution_d, evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, Ndofs);
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	
#ifdef PRINT_SOLUTION
	// Print solution
	printHostSolutionInt(bridge.subdomainOfDOFs, matrix.Ndofs, N);	
	printDeviceSolutionInt(iterationLevel_d, matrix.Ndofs, N);	
#endif
	printf("The number of subdomains are %d\n", bridge.numSubdomains);
	
	/*************** STAGE 3: LOWER PYRAMIDAL STAGE ************************/
	
	printf("==================== PERFORMING LOWER PYRAMIDAL PARTITIONING =========================\n");
	
	// INITIALIZE
	meshPartitionForStage lowerPyramidal;
	numExpansionSteps += (nPerSub-1)/2;
	numJacobiSteps += (nPerSub-1)/2;
	
	// HOST
	constructSeedsHost(lowerPyramidal, N, nSub, 2);
	seedsExpandIntoSubdomains(lowerPyramidal, matrix, iterationLevel, numExpansionSteps);
	createHaloRegions(lowerPyramidal, matrix);
	determineIterationLevelPerDOF(lowerPyramidal, matrix, iterationLevel, 0, numJacobiSteps);
	createTerritoriesHost(lowerPyramidal);
	constructLocalMatricesHost(lowerPyramidal, matrix);
	
	// DEVICE
	meshPartitionForStageDevice lowerPyramidal_d;
	allocatePartitionDevice(lowerPyramidal_d, lowerPyramidal, Ndofs);
	copyPartitionDevice(lowerPyramidal_d, lowerPyramidal, Ndofs);
	
	// JACOBI
	determineSharedMemoryAllocation(lowerPyramidal);
	stageAdvanceJacobiPerformance<<<lowerPyramidal.numSubdomains, 512, lowerPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, lowerPyramidal_d, 0, numJacobiSteps);
	
	// POSTPROCESSING	
	assembleSolutionFromBuffers<<<numBlocks, threadsPerBlock>>>(solution_d, evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, Ndofs);
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	
#ifdef PRINT_SOLUTION
	// Print solution
	printHostSolutionInt(lowerPyramidal.subdomainOfDOFs, matrix.Ndofs, N);	
	printDeviceSolutionInt(iterationLevel_d, matrix.Ndofs, N);	
#endif
	printf("The number of subdomains are %d\n", lowerPyramidal.numSubdomains);
	
	/*************** STAGE 4: DUAL BRIDGE STAGE ************************/
	
	printf("==================== PERFORMING DUAL BRIDGE PARTITIONING =========================\n");

	// INITIALIZE
	meshPartitionForStage dualBridge;

	// HOST
	constructSeedsHost(dualBridge, N, nSub, 3);
	seedsExpandIntoSubdomains(dualBridge, matrix, iterationLevel, numExpansionSteps);
	createHaloRegions(dualBridge, matrix);
	determineIterationLevelPerDOF(dualBridge, matrix, iterationLevel, 0, numJacobiSteps);
	createTerritoriesHost(dualBridge);
	constructLocalMatricesHost(dualBridge, matrix);
	
	// DEVICE
	meshPartitionForStageDevice dualBridge_d;
	allocatePartitionDevice(dualBridge_d, dualBridge, Ndofs);
	copyPartitionDevice(dualBridge_d, dualBridge, Ndofs);
	
	// JACOBI
	determineSharedMemoryAllocation(dualBridge);
	stageAdvanceJacobiPerformance<<<dualBridge.numSubdomains, 512, dualBridge.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, dualBridge_d, 0, numJacobiSteps);
	
	// POSTPROCESSING	
	assembleSolutionFromBuffers<<<numBlocks, threadsPerBlock>>>(solution_d, evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, Ndofs);
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	
#ifdef PRINT_SOLUTION
	// Print solution
	printHostSolutionInt(dualBridge.subdomainOfDOFs, matrix.Ndofs, N);	
	printDeviceSolutionInt(iterationLevel_d, matrix.Ndofs, N);	
#endif
	printf("The number of subdomains are %d\n", dualBridge.numSubdomains);
	
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
	float residual, residualInit;
	residual = computeL2Residual(du0_d, matrix_d);
	residualInit = residual;
	printf("The initial residual is %f\n", residual);
	
	// Initialize solution and residual norm variables
	float * solutionGM = new float[matrix.Ndofs];
	float residualGM;

	// Number of total Jacobi iterations to perform
	uint32_t numIterations = ((nPerSub-1)/2)*2 * numSweptCycles; 

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
	
	// Initial L2 residual
	if (numIterations % 2 == 0) { 
		residual = computeL2Residual(du0_d, matrix_d);
	}
	else if (numIterations % 2 == 1) { 
		residual = computeL2Residual(du1_d, matrix_d);
	}
	residualGM = residual;

#ifdef PRINT_SOLUTION
	// Print solution
	if (numIterations % 2 == 0) { 
		// printDeviceSolutionFloat(du0_d, Ndofs, N);
		solutionGM = du0_d;
	}
	else if (numIterations % 2 == 1) { 
		// printDeviceSolutionFloat(du1_d, Ndofs, N);
		solutionGM = du1_d;
	}
#endif
	
	/*************** SUBDOMAIN CONSTRUCTION COMPLETE - PERFORM ACTUAL ITERATIONS ON GPU ************************/
	
	// Initialize containers for solution and iteration variables 
	threadsPerBlock = 128;
	numBlocks = ceil((float)matrix.Ndofs / threadsPerBlock);
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(evenSolutionBuffer_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(oddSolutionBuffer_d, matrix.Ndofs);	
    initializeToZerosInt(iterationLevel, matrix.Ndofs);
    cudaMemcpy(iterationLevel_d, iterationLevel, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
	uint32_t minJacobiIters = 0;
	uint32_t maxJacobiIters = 0;
	uint32_t maxJacobiShift = 0;

	// Initialize CUDA Event timers
	float time_total_1 = 0;
	float time_total_2 = 0;
	float time_total_3 = 0;
	float time_total_4 = 0;
	float time_total_5 = 0;	
	float time_stage_1, time_stage_2, time_stage_3, time_stage_4;	
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
	
	printf("======================= CYCLE START =========================================\n");

	for (int sweptIteration = 0; sweptIteration < numSweptCycles; sweptIteration++) {

		// Print cycle number
		printf("CYCLE %d\n", sweptIteration);

		if (sweptIteration > 0) {
			maxJacobiShift += 14;
		}

		// Set number of Jacobi iterations for first two stages
		maxJacobiIters += (nPerSub-1)/2;

		// STAGE 1: UPPER PYRAMIDAL
		cudaEventRecord(start_1, 0);
		stageAdvanceJacobiPerformance<<<upperPyramidal.numSubdomains, 256, upperPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, upperPyramidal_d, minJacobiIters, maxJacobiIters, maxJacobiShift);
		cudaEventRecord(stop_1, 0);
		cudaEventSynchronize(stop_1);
		cudaEventElapsedTime(&time_stage_1, start_1, stop_1);
		time_total_1 += time_stage_1;
		
		// Set lower bound for Jacobi iterations
		if (sweptIteration > 0) {
			minJacobiIters += (nPerSub-1)/2;
		}

		// STAGE 2: BRIDGE STAGE
		cudaEventRecord(start_2, 0);
		stageAdvanceJacobiPerformance<<<bridge.numSubdomains, 256, bridge.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, bridge_d, minJacobiIters, maxJacobiIters, maxJacobiShift);
		cudaEventRecord(stop_2, 0);
		cudaEventSynchronize(stop_2);
		cudaEventElapsedTime(&time_stage_2, start_2, stop_2);
		time_total_2 += time_stage_2;

		// Set number of Jacobi iterations for second two stages
		maxJacobiIters += (nPerSub-1)/2;
		
		// STAGE 3: LOWER PYRAMIDAL STAGE
		cudaEventRecord(start_3, 0);
		stageAdvanceJacobiPerformance<<<lowerPyramidal.numSubdomains, 256, lowerPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, lowerPyramidal_d, minJacobiIters, maxJacobiIters, maxJacobiShift);
		cudaEventRecord(stop_3, 0);
		cudaEventSynchronize(stop_3);
		cudaEventElapsedTime(&time_stage_3, start_3, stop_3);
		time_total_3 += time_stage_3;
		
		// Set lower bound for Jacobi iterations
		minJacobiIters += (nPerSub-1)/2;

		// STAGE 4: DUAL BRIDGE STAGE
		cudaEventRecord(start_4, 0);
		stageAdvanceJacobiPerformance<<<dualBridge.numSubdomains, 256, dualBridge.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, dualBridge_d, minJacobiIters, maxJacobiIters, maxJacobiShift);
		cudaEventRecord(stop_4, 0);
		cudaEventSynchronize(stop_4);
		cudaEventElapsedTime(&time_stage_4, start_4, stop_4);
		time_total_4 += time_stage_4;

	}

	// Set number of Jacobi iterations for final fill-in stage
	maxJacobiShift += (nPerSub-1)/2;
	bool finalStage = true;
	
	// FINAL STAGE	
	cudaEventRecord(start_5, 0);
	stageAdvanceJacobiPerformance<<<upperPyramidal.numSubdomains, 256, upperPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, upperPyramidal_d, minJacobiIters, maxJacobiIters, maxJacobiShift, finalStage);
	cudaEventRecord(stop_5, 0);
	cudaEventSynchronize(stop_5);
	cudaEventElapsedTime(&time_total_5, start_5, stop_5);

	// INFORMATION
	
	printf("==================== FINAL INFORMATION =========================\n");
	
	// Compute L2 residual
	assembleSolutionFromBuffers<<<numBlocks, threadsPerBlock>>>(solution_d, evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, Ndofs);
	residual = computeL2Residual(solution_d, matrix_d);
	printf("====================================RESIDUAL====================================================\n");
	printf("Global: The initial residual was %f while the final residual is %f\n", residualInit, residualGM);
	printf("Swept: The initial residual was %f while the final residual is %f\n", residualInit, residual);

	// Print times for global and shared memory to complete
	float sweptTime = time_total_1 + time_total_2 + time_total_3 + time_total_4 + time_total_5;
	printf("===================================TIME INFORMATION=============================================\n");
	printf("Global: Elapsed time in ms %f\n", globalTime);
	printf("Swept: Elapsed time in ms %f\n", sweptTime);
	printf("Time for upper pyramidal is %f\n", time_total_1);
	printf("Time for bridge is %f\n", time_total_2);
	printf("Time for lower pyramidal is %f\n", time_total_3);
	printf("Time for dual bridge is %f\n", time_total_4);
	printf("Time for final step is %f\n", time_total_5);
	printf("Speedup is %f\n", globalTime / sweptTime);

}

