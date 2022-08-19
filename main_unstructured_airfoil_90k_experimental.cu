using namespace std; 
#include "inttypes.h"
#include <vector>
#include <set>
#include <map>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "initialize.h"
#include "structs.h"
#include "helper.h"
#include "mesh.h"
#include "readFromFile.h"
#include "seeds.h"
#include "subdomains.h"
#include "toDevice.h"
#include "global.h"
#include "swept_update_jacobi.h"
#include "swept_update_jacobi_solutionOnly.h"
#include "residual.h"
#include "matrix.h"
#include "print.h"
#include "debug.h"

//# define PRINT_SOLUTION
# define SAVE_SOLUTION

int main (int argc, char * argv[]) 
{
	// Define number of dofs for the mesh
	uint32_t Ndofs = 91936; // 5876; // this is for the coarse airfoil mesh condensed
	uint32_t numSweptCycles = 10000;
	
	// Define the linear system Ax = b

	// Initialize the linear system and allocate matrix data structures (indexPtr, nodeNeighbors, offdiags) 
	linearSystem matrix;
	matrix.Ndofs = Ndofs;
	initializeAndLoadMatrixFromDirectory(matrix, "Unstructured_Mesh/Airfoil_Mesh_90k/Matrix_Data");   

	for (int i = 0; i < matrix.Ndofs; i++) {
	//	printf("rhs[%d] = %f\n", i, matrix.rhs[i]);
	}

	// Allocate matrix data structures to the GPU (is this even necessary? - yes! used by global memory solution) 
	linearSystemDevice matrix_d;
  	allocateMatrixDevice(matrix_d, matrix);
    copyMatrixDevice(matrix_d, matrix);
	
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
	uint32_t numIterations = numSweptCycles * 12;

	// Perform global memory iterations
	float globalTime;	
	cudaEvent_t start_g, stop_g;
	cudaEventCreate(&start_g);
	cudaEventCreate(&stop_g);
	cudaEventRecord(start_g, 0); 
    // globalMemorySolve2D(du1_d, du0_d, matrix_d, N, numIterations);
    globalMemorySolveUnstructured(du1_d, du0_d, matrix_d, numIterations);
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
	printf("GLOBAL MEMORY ITERATIONS: The initial residual is %f but the final residual after %d iterations is %f\n", residualInit, numIterations, residual);

	// Print solution
	if (numIterations % 2 == 0) { 
		solutionGM = du0_d;
	}
	else if (numIterations % 2 == 1) { 
		solutionGM = du1_d;
	}
	printf("Number of Iterations = %d\n", numIterations);

	// SHARED MEMORY START 
	
	printf("==================== SHARED MEMORY ALGORITHM =========================\n");
	
	// Initialize iteration level
	uint32_t * iterationLevel, * iterationLevel_d, * iterationLevelOutput_d;
	iterationLevel = new uint32_t[Ndofs];
	initializeToZerosInt(iterationLevel, Ndofs);
	// Iteration Level
    cudaMalloc(&iterationLevel_d, sizeof(uint32_t) * Ndofs);
	cudaMemcpy(iterationLevel_d, iterationLevel, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
    cudaMalloc(&iterationLevelOutput_d, sizeof(uint32_t) * Ndofs);
	cudaMemcpy(iterationLevelOutput_d, iterationLevel, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
	
	float * evenSolutionBuffer_d, * oddSolutionBuffer_d, * solution_d;
	float * evenSolutionBufferOutput_d, * oddSolutionBufferOutput_d;
	cudaMalloc(&evenSolutionBuffer_d, sizeof(float) * matrix.Ndofs);
	cudaMalloc(&oddSolutionBuffer_d, sizeof(float) * matrix.Ndofs);
	cudaMalloc(&evenSolutionBufferOutput_d, sizeof(float) * matrix.Ndofs);
	cudaMalloc(&oddSolutionBufferOutput_d, sizeof(float) * matrix.Ndofs);
	cudaMalloc(&solution_d, sizeof(float) * matrix.Ndofs);
	uint32_t threadsPerBlock = 256;
	uint32_t numBlocks = ceil((float)matrix.Ndofs / threadsPerBlock);
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(evenSolutionBuffer_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(oddSolutionBuffer_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(evenSolutionBufferOutput_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(oddSolutionBufferOutput_d, matrix.Ndofs);	
	
	// Number of Jacobi iterations to perform in half swept cycle
	uint32_t numJacobiIters = 6;
	
	// DIRECTORY CONTAINING PARTITION AND LEVEL INFORMATION
	std::string PARENT_DIRECTORY = "Unstructured_Mesh/Airfoil_Mesh_90k/";

	// Temporary pointers for swapping	
	float * tmp;
	uint32_t * tmpInt;
	
	/*************** STAGE 1: UPPER PYRAMIDAL STAGE ************************/

	printf("==================== PERFORMING UPPER PYRAMIDAL PARTITIONING =========================\n");
	
	// INITIALIZE
	meshPartitionForStage upperPyramidal;
	upperPyramidal.numSubdomains = 250; // 25

	// SUBDOMAINS
	readSubdomainAndIterationFromFile2(upperPyramidal, PARENT_DIRECTORY, 0);	

	// HOST
	createHaloRegions(upperPyramidal, matrix);
	createTerritoriesHost(upperPyramidal);
	constructLocalMatricesHost(upperPyramidal, matrix);
	
	// DEVICE
	meshPartitionForStageDevice upperPyramidal_d;
	allocatePartitionDevice(upperPyramidal_d, upperPyramidal, Ndofs);
	copyPartitionDevice(upperPyramidal_d, upperPyramidal, Ndofs);
	
	// JACOBI
	// V1 kernel (non-overlapping)
	determineSharedMemoryAllocationSolutionOnly(upperPyramidal);
	// determineSharedMemoryAllocationSolutionOnly(upperPyramidal);
	stageAdvanceJacobiPerformanceSolutionOnly<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, upperPyramidal_d, 0, numJacobiIters);
	//determineSharedMemoryAllocationSolutionOnly(upperPyramidal);
	//stageAdvanceJacobiPerformanceSolutionOnly<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, upperPyramidal_d, 0, numJacobiIters);
	// V2 kernel (overlapping)
	// stageAdvanceJacobiPerformanceV2Overlap<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(evenSolutionBufferOutput_d, oddSolutionBufferOutput_d, evenSolutionBuffer_d, oddSolutionBuffer_d,  iterationLevel_d, iterationLevelOutput_d, upperPyramidal_d, 0, numJacobiIters, Ndofs);
	// tmp = evenSolutionBuffer_d; evenSolutionBuffer_d = evenSolutionBufferOutput_d; evenSolutionBufferOutput_d = tmp;
	// tmp = oddSolutionBuffer_d; oddSolutionBuffer_d = oddSolutionBufferOutput_d; oddSolutionBufferOutput_d = tmp;
	// tmpInt = iterationLevel_d; iterationLevel_d = iterationLevelOutput_d; iterationLevelOutput_d = tmpInt;

	// POSTPROCESSING	
	assembleSolutionFromBuffers<<<numBlocks, threadsPerBlock>>>(solution_d, evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, Ndofs);
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	
	uint32_t * iterationLevelUpperPyramidal = new uint32_t[Ndofs];
	cudaMemcpy(iterationLevelUpperPyramidal, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	

#ifdef PRINT_SOLUTION
	printf("================NUMBER OF ITERATIONS PERFORMED IN SHARED==============\n");
	printDeviceArrayInt(iterationLevel_d, Ndofs);
	printf("\n================SIMILARITY TO GLOBAL==============\n");
	printDeviceSimilarity1D(solution_d, solutionGM, 1e-6, Ndofs);
	printGlobalAndSharedMatchDevice(solution_d, solutionGM, iterationLevel_d, numIterations, Ndofs);
#endif
	// printGlobalAndSharedMatchDevice(solution_d, solutionGM, iterationLevel_d, numIterations, Ndofs);

#ifdef SAVE_SOLUTION
	// Save iteration level	
  	ofstream iterationLevelFile;
	iterationLevelFile.open(PARENT_DIRECTORY + "iteration_output_1.txt");
	for (int i = 0; i < matrix.Ndofs; i++) {
  		iterationLevelFile << iterationLevel[i];
  		iterationLevelFile << "\n";
	}
  	iterationLevelFile.close();
#endif

	cudaError err;
	err = cudaDeviceSynchronize();
	printf("%d\n", err);
	if (cudaSuccess != err) {
		fprintf(stderr, "failed at %s\n", cudaGetErrorString(err));
		exit(-1);
	}
	
    residual = computeL2Residual(solution_d, matrix_d);
	printf("The residual after stage 1 is %f\n", residual);
	
	/*************** STAGE 2: BRIDGE STAGE ************************/
	
	printf("==================== PERFORMING BRIDGE PARTITIONING =========================\n");

	// INITIALIZE
	meshPartitionForStage bridge;
	bridge.numSubdomains = 397; // 397; // 35

	// SUBDOMAINS
	readSubdomainAndIterationFromFile2(bridge, PARENT_DIRECTORY, 1);	
	
	// HOST
	createHaloRegions(bridge, matrix);
	createTerritoriesHost(bridge);
	constructLocalMatricesHost(bridge, matrix);
	
	// DEVICE
	meshPartitionForStageDevice bridge_d;
	allocatePartitionDevice(bridge_d, bridge, Ndofs);
	copyPartitionDevice(bridge_d, bridge, Ndofs);
	
	// JACOBI
	determineSharedMemoryAllocationSolutionOnly(bridge);
	// V1 kernel (non-overlapping)
	// stageAdvanceJacobiPerformance<<<bridge.numSubdomains, 512, bridge.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, bridge_d, 0, numJacobiIters);
	// V2 kernel (overlapping)
	copySolutionToOutput<<<bridge.numSubdomains, 512>>>(evenSolutionBufferOutput_d, oddSolutionBufferOutput_d, evenSolutionBuffer_d, oddSolutionBuffer_d,  iterationLevel_d, iterationLevelOutput_d, Ndofs);
	stageAdvanceJacobiPerformanceV2OverlapExperimentalSolutionOnly<<<bridge.numSubdomains, 512, bridge.sharedMemorySize>>>(evenSolutionBufferOutput_d, oddSolutionBufferOutput_d, evenSolutionBuffer_d, oddSolutionBuffer_d,  iterationLevel_d, iterationLevelOutput_d, bridge_d, 0, numJacobiIters, Ndofs);
	cudaDeviceSynchronize();
	tmp = evenSolutionBuffer_d; evenSolutionBuffer_d = evenSolutionBufferOutput_d; evenSolutionBufferOutput_d = tmp;
	tmp = oddSolutionBuffer_d; oddSolutionBuffer_d = oddSolutionBufferOutput_d; oddSolutionBufferOutput_d = tmp;
	tmpInt = iterationLevel_d; iterationLevel_d = iterationLevelOutput_d; iterationLevelOutput_d = tmpInt;
	
	// POSTPROCESSING	
	cudaDeviceSynchronize();
	assembleSolutionFromBuffers<<<numBlocks, threadsPerBlock>>>(solution_d, evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, Ndofs);
	cudaDeviceSynchronize();
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	
	cudaDeviceSynchronize();
	uint32_t * iterationLevelBridge = new uint32_t[Ndofs];
	cudaMemcpy(iterationLevelBridge, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	

#ifdef PRINT_SOLUTION
	printf("================NUMBER OF ITERATIONS PERFORMED IN SHARED==============\n");
	printDeviceArrayInt(iterationLevel_d, Ndofs);
	printf("\n================SIMILARITY TO GLOBAL==============\n");
	printDeviceSimilarity1D(solution_d, solutionGM, 1e-6, Ndofs);
	printGlobalAndSharedMatchDevice(solution_d, solutionGM, iterationLevel_d, numIterations, Ndofs);
#endif
	printGlobalAndSharedMatchDevice(solution_d, solutionGM, iterationLevel_d, numIterations, Ndofs);

#ifdef SAVE_SOLUTION
	// Save iteration level	
	iterationLevelFile.open(PARENT_DIRECTORY + "iteration_output_2.txt");
	for (int i = 0; i < matrix.Ndofs; i++) {
  		iterationLevelFile << iterationLevel[i];
  		iterationLevelFile << "\n";
	}
  	iterationLevelFile.close();
#endif
		
	err = cudaDeviceSynchronize();
	printf("%d\n", err);
	if (cudaSuccess != err) {
		fprintf(stderr, "failed at %s\n", cudaGetErrorString(err));
		exit(-1);
	}
 	
	residual = computeL2Residual(solution_d, matrix_d);
	printf("The residual after stage 2 is %f\n", residual);
	
	/*************** STAGE 3: LOWER PYRAMIDAL STAGE ************************/
	
	printf("==================== PERFORMING LOWER PYRAMIDAL PARTITIONING =========================\n");

	// INITIALIZE
	meshPartitionForStage lowerPyramidal;
	lowerPyramidal.numSubdomains = 309; // 22;

	// SUBDOMAINS
	readSubdomainAndIterationFromFile2(lowerPyramidal, PARENT_DIRECTORY, 2);	
	
	// HOST
	createHaloRegions(lowerPyramidal, matrix);
	createTerritoriesHost(lowerPyramidal);
	constructLocalMatricesHost(lowerPyramidal, matrix);

	// DEVICE
	meshPartitionForStageDevice lowerPyramidal_d;
	allocatePartitionDevice(lowerPyramidal_d, lowerPyramidal, Ndofs);
	copyPartitionDevice(lowerPyramidal_d, lowerPyramidal, Ndofs);
	
	// JACOBI
	determineSharedMemoryAllocationSolutionOnly(lowerPyramidal);
	// V1 kernel (non-overlapping)
	stageAdvanceJacobiPerformanceSolutionOnly<<<lowerPyramidal.numSubdomains, 512, lowerPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, lowerPyramidal_d, 0, numJacobiIters); 
	// V2 kernel (overlapping)
	// stageAdvanceJacobiPerformanceV2Overlap<<<lowerPyramidal.numSubdomains, 512, lowerPyramidal.sharedMemorySize>>>(evenSolutionBufferOutput_d, oddSolutionBufferOutput_d, evenSolutionBuffer_d, oddSolutionBuffer_d,  iterationLevel_d, iterationLevelOutput_d, lowerPyramidal_d, 0, numJacobiIters, Ndofs);
	// tmp = evenSolutionBuffer_d; evenSolutionBuffer_d = evenSolutionBufferOutput_d; evenSolutionBufferOutput_d = tmp;
	// tmp = oddSolutionBuffer_d; oddSolutionBuffer_d = oddSolutionBufferOutput_d; oddSolutionBufferOutput_d = tmp;
	// tmpInt = iterationLevel_d; iterationLevel_d = iterationLevelOutput_d; iterationLevelOutput_d = tmpInt;

	// POSTPROCESSING	
	assembleSolutionFromBuffers<<<numBlocks, threadsPerBlock>>>(solution_d, evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, Ndofs);
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	

#ifdef PRINT_SOLUTION
	printf("================NUMBER OF ITERATIONS PERFORMED IN SHARED==============\n");
	printDeviceArrayInt(iterationLevel_d, Ndofs);
	printf("\n================SIMILARITY TO GLOBAL==============\n");
	printDeviceSimilarity1D(solution_d, solutionGM, 1e-6, Ndofs);
	printGlobalAndSharedMatchDevice(solution_d, solutionGM, iterationLevel_d, numIterations, Ndofs);
#endif
	// printGlobalAndSharedMatchDevice(solution_d, solutionGM, iterationLevel_d, numIterations, Ndofs);

#ifdef SAVE_SOLUTION
	// Save iteration level	
	iterationLevelFile.open(PARENT_DIRECTORY + "iteration_output_3.txt");
	for (int i = 0; i < matrix.Ndofs; i++) {
		// printf("Iteration Level[%d] = %d\n", i, iterationLevel[i]);
  		iterationLevelFile << iterationLevel[i];
  		iterationLevelFile << "\n";
	}
  	iterationLevelFile.close();
#endif

	err = cudaDeviceSynchronize();
	printf("%d\n", err);
	if (cudaSuccess != err) {
		fprintf(stderr, "failed at %s\n", cudaGetErrorString(err));
		exit(-1);
	}

	residual = computeL2Residual(solution_d, matrix_d);
	printf("The residual after stage 3 is %f\n", residual);
	
	/*************** STAGE 4: DUAL BRIDGE STAGE ************************/
	
	printf("==================== PERFORMING DUAL BRIDGE PARTITIONING =========================\n");
	
	// INITIALIZE
	meshPartitionForStage dualBridge;
	dualBridge.numSubdomains = 397; // 35;

	// SUBDOMAINS
	readSubdomainAndIterationFromFile2(dualBridge, PARENT_DIRECTORY, 3);

	// HOST
	createHaloRegions(dualBridge, matrix);
	createTerritoriesHost(dualBridge);
	constructLocalMatricesHost(dualBridge, matrix);
	
	// DEVICE
	meshPartitionForStageDevice dualBridge_d;
	allocatePartitionDevice(dualBridge_d, dualBridge, Ndofs);
	copyPartitionDevice(dualBridge_d, dualBridge, Ndofs);
	
	// JACOBI
	// Perform Jacobi Iteration kernel call
	determineSharedMemoryAllocationSolutionOnly(dualBridge);
	// V1 kernel (non-overlapping)
	// stageAdvanceJacobiPerformance<<<dualBridge.numSubdomains, 512, dualBridge.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, dualBridge_d, 0, 2*numJacobiIters); 
	// V2 kernel (overlapping)
	copySolutionToOutput<<<dualBridge.numSubdomains, 512>>>(evenSolutionBufferOutput_d, oddSolutionBufferOutput_d, evenSolutionBuffer_d, oddSolutionBuffer_d,  iterationLevel_d, iterationLevelOutput_d, Ndofs);
	stageAdvanceJacobiPerformanceV2OverlapExperimentalSolutionOnly<<<dualBridge.numSubdomains, 512, dualBridge.sharedMemorySize>>>(evenSolutionBufferOutput_d, oddSolutionBufferOutput_d, evenSolutionBuffer_d, oddSolutionBuffer_d,  iterationLevel_d, iterationLevelOutput_d, dualBridge_d, 0, numJacobiIters, Ndofs);
	tmp = evenSolutionBuffer_d; evenSolutionBuffer_d = evenSolutionBufferOutput_d; evenSolutionBufferOutput_d = tmp;
	tmp = oddSolutionBuffer_d; oddSolutionBuffer_d = oddSolutionBufferOutput_d; oddSolutionBufferOutput_d = tmp;
	tmpInt = iterationLevel_d; iterationLevel_d = iterationLevelOutput_d; iterationLevelOutput_d = tmpInt;
	
	// POSTPROCESSING	
	assembleSolutionFromBuffers<<<numBlocks, threadsPerBlock>>>(solution_d, evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, Ndofs);
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	

#ifdef PRINT_SOLUTION
	printf("================NUMBER OF ITERATIONS PERFORMED IN SHARED==============\n");
	printDeviceArrayInt(iterationLevel_d, Ndofs);
	printf("\n================SIMILARITY TO GLOBAL==============\n");
	printDeviceSimilarity1D(solution_d, solutionGM, 1e-6, Ndofs);
	printGlobalAndSharedMatchDevice(solution_d, solutionGM, iterationLevel_d, numIterations, Ndofs);
#endif
	// printGlobalAndSharedMatchDevice(solution_d, solutionGM, iterationLevel_d, numIterations, Ndofs);

#ifdef SAVE_SOLUTION
	// Save iteration level	
	iterationLevelFile.open(PARENT_DIRECTORY + "iteration_output_4.txt");
	for (int i = 0; i < matrix.Ndofs; i++) {
		// printf("Iteration Level[%d] = %d\n", i, iterationLevel[i]);
  		iterationLevelFile << iterationLevel[i];
  		iterationLevelFile << "\n";
	}
  	iterationLevelFile.close();
#endif

	err = cudaDeviceSynchronize();
	printf("%d\n", err);
	if (cudaSuccess != err) {
		fprintf(stderr, "failed at %s\n", cudaGetErrorString(err));
		exit(-1);
	}

	residual = computeL2Residual(solution_d, matrix_d);
	printf("The residual after stage 4 is %f\n", residual);
	
	/*************** SUBDOMAIN CONSTRUCTION COMPLETE - PERFORM ACTUAL ITERATIONS ON GPU ************************/
	
	// Initialize containers for solution and iteration variables
	threadsPerBlock = 128;
	numBlocks = ceil((float)matrix.Ndofs / threadsPerBlock);
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(evenSolutionBuffer_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(oddSolutionBuffer_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(evenSolutionBufferOutput_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(oddSolutionBufferOutput_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(solution_d, matrix.Ndofs);	
    initializeToZerosInt(iterationLevel, matrix.Ndofs);
    cudaMemcpy(iterationLevel_d, iterationLevel, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
    cudaMemcpy(iterationLevelOutput_d, iterationLevel, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
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
			maxJacobiShift += 12;
		}

		// Set number of Jacobi iterations for first two stages
		maxJacobiIters += 6;

		// STAGE 1: UPPER PYRAMIDAL
		cudaEventRecord(start_1, 0);
		// V1 kernel (non-overlapping)
		stageAdvanceJacobiPerformanceSolutionOnly<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, upperPyramidal_d, minJacobiIters, maxJacobiIters, maxJacobiShift);
		// stageAdvanceJacobiPerformanceSolutionOnly<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, upperPyramidal_d, minJacobiIters, maxJacobiIters, maxJacobiShift);
		// V2 kernel (overlapping)
		// copySolutionToOutput<<<upperPyramidal.numSubdomains, 512>>>(evenSolutionBufferOutput_d, oddSolutionBufferOutput_d, evenSolutionBuffer_d, oddSolutionBuffer_d,  iterationLevel_d, iterationLevelOutput_d, Ndofs);
		// stageAdvanceJacobiPerformanceV2OverlapExperimental<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(evenSolutionBufferOutput_d, oddSolutionBufferOutput_d, evenSolutionBuffer_d, oddSolutionBuffer_d,  iterationLevel_d, iterationLevelOutput_d, upperPyramidal_d, 0, numJacobiIters, Ndofs);
		// tmp = evenSolutionBuffer_d; evenSolutionBuffer_d = evenSolutionBufferOutput_d; evenSolutionBufferOutput_d = tmp;
		// tmp = oddSolutionBuffer_d; oddSolutionBuffer_d = oddSolutionBufferOutput_d; oddSolutionBufferOutput_d = tmp;
		// tmpInt = iterationLevel_d; iterationLevel_d = iterationLevelOutput_d; iterationLevelOutput_d = tmpInt;
		//
		cudaEventRecord(stop_1, 0);
		cudaEventSynchronize(stop_1);
		cudaEventElapsedTime(&time_stage_1, start_1, stop_1);
		time_total_1 += time_stage_1;
		//printf("STAGE 1\n");
		//printDeviceArrayInt(iterationLevel_d, Ndofs);

		if (sweptIteration > 0) {
			minJacobiIters += 6;
		}
	
		// STAGE 2: BRIDGE STAGE
		cudaEventRecord(start_2, 0);
		// V2 kernel (overlapping)
		copySolutionToOutput<<<numBlocks, 256>>>(evenSolutionBufferOutput_d, oddSolutionBufferOutput_d, evenSolutionBuffer_d, oddSolutionBuffer_d,  iterationLevel_d, iterationLevelOutput_d, Ndofs);
		stageAdvanceJacobiPerformanceV2OverlapExperimentalSolutionOnly<<<bridge.numSubdomains, 512, bridge.sharedMemorySize>>>(evenSolutionBufferOutput_d, oddSolutionBufferOutput_d, evenSolutionBuffer_d, oddSolutionBuffer_d,  iterationLevel_d, iterationLevelOutput_d, bridge_d, minJacobiIters, maxJacobiIters, Ndofs, maxJacobiShift);
		tmp = evenSolutionBuffer_d; evenSolutionBuffer_d = evenSolutionBufferOutput_d; evenSolutionBufferOutput_d = tmp;
		tmp = oddSolutionBuffer_d; oddSolutionBuffer_d = oddSolutionBufferOutput_d; oddSolutionBufferOutput_d = tmp;
		tmpInt = iterationLevel_d; iterationLevel_d = iterationLevelOutput_d; iterationLevelOutput_d = tmpInt;
		//
		cudaEventRecord(stop_2, 0);
		cudaEventSynchronize(stop_2);
		cudaEventElapsedTime(&time_stage_2, start_2, stop_2);
		time_total_2 += time_stage_2;
		//printf("STAGE 2\n");
		//printDeviceArrayInt(iterationLevel_d, Ndofs);

		// Set number of Jacobi iterations for second two stages
		maxJacobiIters += 6;

		// STAGE 3: LOWER PYRAMIDAL
		cudaEventRecord(start_3, 0);
		// V1 kernel (non-overlapping)
		// stageAdvanceJacobiPerformance<<<lowerPyramidal.numSubdomains, 512, lowerPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, lowerPyramidal_d, minJacobiIters, maxJacobiIters, maxJacobiShift);
		stageAdvanceJacobiPerformanceSolutionOnly<<<lowerPyramidal.numSubdomains, 512, lowerPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, lowerPyramidal_d, minJacobiIters, maxJacobiIters, maxJacobiShift);
		// V2 kernel (overlapping)
		// stageAdvanceJacobiPerformanceV2Overlap<<<lowerPyramidal.numSubdomains, 512, lowerPyramidal.sharedMemorySize>>>(evenSolutionBufferOutput_d, oddSolutionBufferOutput_d, evenSolutionBuffer_d, oddSolutionBuffer_d,  iterationLevel_d, iterationLevelOutput_d, lowerPyramidal_d, minJacobiIters, maxJacobiIters, Ndofs);
		// tmp = evenSolutionBuffer_d; evenSolutionBuffer_d = evenSolutionBufferOutput_d; evenSolutionBufferOutput_d = tmp;
		// tmp = oddSolutionBuffer_d; oddSolutionBuffer_d = oddSolutionBufferOutput_d; oddSolutionBufferOutput_d = tmp;
		// tmpInt = iterationLevel_d; iterationLevel_d = iterationLevelOutput_d; iterationLevelOutput_d = tmpInt;

		//
		cudaEventRecord(stop_3, 0);
		cudaEventSynchronize(stop_3);
		cudaEventElapsedTime(&time_stage_3, start_3, stop_3);
		time_total_3 += time_stage_3;
		
		// Set number of Jacobi iterations for second two stages
		minJacobiIters += 6;
		
		// Dual Bridge
		cudaEventRecord(start_4, 0);
		// V2 kernel (overlapping)
		copySolutionToOutput<<<numBlocks, 256>>>(evenSolutionBufferOutput_d, oddSolutionBufferOutput_d, evenSolutionBuffer_d, oddSolutionBuffer_d,  iterationLevel_d, iterationLevelOutput_d, Ndofs);
		stageAdvanceJacobiPerformanceV2OverlapExperimentalSolutionOnly<<<dualBridge.numSubdomains, 512, dualBridge.sharedMemorySize>>>(evenSolutionBufferOutput_d, oddSolutionBufferOutput_d, evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, iterationLevelOutput_d, dualBridge_d, minJacobiIters, maxJacobiIters, Ndofs, maxJacobiShift);
		tmp = evenSolutionBuffer_d; evenSolutionBuffer_d = evenSolutionBufferOutput_d; evenSolutionBufferOutput_d = tmp;
		tmp = oddSolutionBuffer_d; oddSolutionBuffer_d = oddSolutionBufferOutput_d; oddSolutionBufferOutput_d = tmp;
		tmpInt = iterationLevel_d; iterationLevel_d = iterationLevelOutput_d; iterationLevelOutput_d = tmpInt;
		// 
		cudaEventRecord(stop_4, 0);
		cudaEventSynchronize(stop_4);
		cudaEventElapsedTime(&time_stage_4, start_4, stop_4);
		time_total_4 += time_stage_4;
	
	}

	// Set number of Jacobi iterations for final fill-in stage
	maxJacobiShift += 6;
	bool finalStage = true; 

	// FINAL STAGE
	cudaEventRecord(start_5, 0);
	// V1 kernel (non-overlapping)
	stageAdvanceJacobiPerformanceSolutionOnly<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, upperPyramidal_d, minJacobiIters, maxJacobiIters, maxJacobiShift, finalStage);
	//
	cudaEventRecord(stop_5, 0);
	cudaEventSynchronize(stop_5);
	cudaEventElapsedTime(&time_total_5, start_5, stop_5);

#ifdef SAVE_SOLUTION	
	// Print iteration level
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	
	iterationLevelFile.open(PARENT_DIRECTORY + "iteration_output_final.txt");
	for (int i = 0; i < matrix.Ndofs; i++) {
  		iterationLevelFile << iterationLevel[i];
  		iterationLevelFile << "\n";
	}
  	iterationLevelFile.close();
#endif

	// Print information
	printf("\n==================== FINAL INFORMATION =========================\n");
	assembleSolutionFromBuffers<<<numBlocks, threadsPerBlock>>>(solution_d, evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, Ndofs);
#ifdef PRINT_SOLUTION
	printf("================NUMBER OF ITERATIONS PERFORMED IN SHARED==============\n");
	printDeviceArrayInt(iterationLevel_d, Ndofs);
	printf("\n================SIMILARITY TO GLOBAL==============\n");
	printDeviceSimilarity1D(solution_d, solutionGM, 1e-6, Ndofs);
	printGlobalAndSharedMatchDevice(solution_d, solutionGM, iterationLevel_d, numIterations, Ndofs);
#endif
	// Compute L2 residual
	residual = computeL2Residual(solution_d, matrix_d);
	printf("========================RESIDUAL=====================================================================\n");
	printf("Global: The initial residual was %f while the final residual is %f\n", residualInit, residualGM);
	printf("Swept: The initial residual was %f while the final residual is %f\n", residualInit, residual);

	// Print times for global and shared memory to complete
	float sweptTime = time_total_1 + time_total_2 + time_total_3 + time_total_4 + time_total_5;
	printf("========================TIME INFORMATION==============================================================\n");
	printf("Global: Elapsed time in ms %f\n", globalTime);
	printf("Swept: Elapsed time in microseconds %f\n", sweptTime);
	printf("Time for upper pyramidal is %f\n", time_total_1);
	printf("Time for bridge is %f\n", time_total_2);
	printf("Time for lower pyramidal is %f\n", time_total_3);
	printf("Time for dual bridge is %f\n", time_total_4);
	printf("Time for final step is %f\n", time_total_5);
	printf("Speedup is %f\n", globalTime / sweptTime);
	// printDeviceArrayInt(iterationLevel_d, Ndofs);
	
}

