using namespace std; 
#include "inttypes.h"
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
#include "toDevice.h"
#include "global.h"
#include "swept_update_jacobi.h"
#include "residual.h"
#include "initialize.h"
#include "matrix.h"
#include "print.h"
#include "debug.h"

// # define PRINT_SOLUTION
# define SAVE_SOLUTION

int main (int argc, char * argv[]) 
{
	// Define number of dofs for the mesh
	uint32_t Ndofs = 394; // this is for the coarse airfoil mesh condensed
	uint32_t numSweptCycles = 1;
	
	// Define the linear system Ax = b

	// Initialize the linear system and allocate matrix data structures (indexPtr, nodeNeighbors, offdiags) 
	linearSystem matrix;
	matrix.Ndofs = Ndofs;
	initializeAndLoadMatrixFromDirectory(matrix, "Unstructured_Mesh/Square_Mesh/Matrix_Data");   

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
	uint32_t numIterations = numSweptCycles * 10;

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

#ifdef PRINT_SOLUTION
	// Print solution
	if (numIterations % 2 == 0) { 
		solutionGM = du0_d;
	}
	else if (numIterations % 2 == 1) { 
		solutionGM = du1_d;
	}
	printf("Number of Iterations = %d\n", numIterations);
#endif

	// SHARED MEMORY START 
	
	printf("==================== SHARED MEMORY ALGORITHM =========================\n");
	
	// Initialize iteration level
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
	uint32_t numExpansionSteps = 4;
	uint32_t numJacobiSteps = 5;

	// SEEDS
	set<int> seed1_1 = {233};
	set<int> seed1_2 = {10};
	set<int> seed1_3 = {47, 50};
	set<int> seed1_4 = {355, 369, 375, 381, 385};
	set<int> seed1_5 = {383};
	upperPyramidal.seeds.push_back(seed1_1);
	upperPyramidal.seeds.push_back(seed1_2);
	upperPyramidal.seeds.push_back(seed1_3);
	upperPyramidal.seeds.push_back(seed1_4);
	upperPyramidal.seeds.push_back(seed1_5);
	upperPyramidal.numSubdomains = 5;

	// HOST
	seedsExpandIntoSubdomains(upperPyramidal, matrix, iterationLevel, numExpansionSteps);
	createHaloRegions(upperPyramidal, matrix);
	createTerritoriesHost(upperPyramidal);
	determineMaximumIterationsPerDOF(upperPyramidal, matrix, iterationLevel, 5);
	constructLocalMatricesHost(upperPyramidal, matrix);
	
	// DEVICE
	meshPartitionForStageDevice upperPyramidal_d;
	allocatePartitionDevice(upperPyramidal_d, upperPyramidal, Ndofs);
	copyPartitionDevice(upperPyramidal_d, upperPyramidal, Ndofs);
	
	// JACOBI
	determineSharedMemoryAllocation(upperPyramidal);
	stageAdvanceJacobiPerformance<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, upperPyramidal_d, 0, numJacobiSteps);
	
	// POSTPROCESSING	
	assembleSolutionFromBuffers<<<numBlocks, threadsPerBlock>>>(solution_d, evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d);
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	

#ifdef PRINT_SOLUTION
	printf("================NUMBER OF ITERATIONS PERFORMED IN SHARED==============\n");
	printDeviceArrayInt(iterationLevel_d, Ndofs);
	printf("\n================SIMILARITY TO GLOBAL==============\n");
	printDeviceSimilarity1D(solution_d, solutionGM, 1e-6, Ndofs);
	printGlobalAndSharedMatchDevice(solution_d, solutionGM, iterationLevel_d, numIterations, Ndofs);
#endif

#ifdef SAVE_SOLUTION
	// Save iteration level	
  	ofstream iterationLevelFile;
	iterationLevelFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/iteration_1.txt");
	for (int i = 0; i < matrix.Ndofs; i++) {
  		iterationLevelFile << iterationLevel[i];
  		iterationLevelFile << "\n";
	}
  	iterationLevelFile.close();
	// Save territories	
  	ofstream territoriesFile;
	for (int i = 0; i < upperPyramidal.numSubdomains; i++) {
		if (i == 0) {
			territoriesFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/territories_1_1.txt");
		}
		else if (i == 1) {
			territoriesFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/territories_1_2.txt");
		}
		else if (i == 2) {
			territoriesFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/territories_1_3.txt");
		}
		else if (i == 3) {
			territoriesFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/territories_1_4.txt");
		}
		else if (i == 4) {
			territoriesFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/territories_1_5.txt");
		}
		for (auto dof : upperPyramidal.territoryDOFsInterior[i]) {
  			territoriesFile << dof;
  			territoriesFile << "\n";
		}
		for (auto dof : upperPyramidal.territoryDOFsInteriorExt[i]) {
  			territoriesFile << dof;
  			territoriesFile << "\n";
		}
		// for (auto dof : bridge.territoryDOFsExterior[i]) {
  		//	territoriesFile << dof;
  		//	territoriesFile << "\n";
		//} 
  		territoriesFile.close();
	}
#endif
	
	/*************** STAGE 2: BRIDGE STAGE ************************/
	
	printf("==================== PERFORMING BRIDGE PARTITIONING =========================\n");

	// INITIALIZE
	meshPartitionForStage bridge;

	// SEEDS
	set<int> seed2_1;
	set<int> seed2_2;
	set<int> seed2_3;
	set<int> seed2_4;
	int bridgeSeed;
	int numSeeds_1 = 36;
	int numSeeds_2 = 7;
	int numSeeds_3 = 60;
	int numSeeds_4 = 101;
	std::ifstream bridgeSeedFile1("Unstructured_Mesh/Square_Mesh/Seeds/bridgeSeed_1.txt");	
	std::ifstream bridgeSeedFile2("Unstructured_Mesh/Square_Mesh/Seeds/bridgeSeed_2.txt");	
	std::ifstream bridgeSeedFile3("Unstructured_Mesh/Square_Mesh/Seeds/bridgeSeed_3.txt");	
	std::ifstream bridgeSeedFile4("Unstructured_Mesh/Square_Mesh/Seeds/bridgeSeed_4.txt");	
	// Seed 1	
	for (int i = 0; i < numSeeds_1; i++) {
		bridgeSeedFile1 >> bridgeSeed;
		seed2_1.insert(bridgeSeed);
	}
	bridge.seeds.push_back(seed2_1);
	// Seed 2	
	for (int i = 0; i < numSeeds_2; i++) {
		bridgeSeedFile2 >> bridgeSeed;
		seed2_2.insert(bridgeSeed);
	}
	bridge.seeds.push_back(seed2_2);
	// Seed 3	
	for (int i = 0; i < numSeeds_3; i++) {
		bridgeSeedFile3 >> bridgeSeed;
		seed2_3.insert(bridgeSeed);
	}
	bridge.seeds.push_back(seed2_3);
	// Seed 4	
	for (int i = 0; i < numSeeds_4; i++) {
		bridgeSeedFile4 >> bridgeSeed;
		seed2_4.insert(bridgeSeed);
	}
	bridge.seeds.push_back(seed2_4);
	bridge.numSubdomains = 4;
	
	// HOST
	seedsExpandIntoSubdomainsSimple(bridge, matrix, iterationLevel, numExpansionSteps);
	//numExpansionSteps = 1;
	// seedsExpandIntoSubdomains(bridge, matrix, iterationLevel, numExpansionSteps);
	createHaloRegions(bridge, matrix);
	createTerritoriesHost(bridge);
	determineMaximumIterationsPerDOF(bridge, matrix, iterationLevel, 5);
	constructLocalMatricesHost(bridge, matrix);
	// printHostArrayInt(bridge.maximumIterations, bridge.territoryIndexPtrInteriorExt[bridge.numSubdomains]);
	
	// DEVICE
	meshPartitionForStageDevice bridge_d;
	allocatePartitionDevice(bridge_d, bridge, Ndofs);
	copyPartitionDevice(bridge_d, bridge, Ndofs);
	// printDeviceArrayInt(bridge_d.maximumIterations_d, bridge.territoryIndexPtrInteriorExt[bridge.numSubdomains]);
	
	// JACOBI
	determineSharedMemoryAllocation(bridge);
	stageAdvanceJacobiPerformance<<<bridge.numSubdomains, 512, bridge.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, bridge_d, 0, numJacobiSteps);

	// POSTPROCESSING	
	assembleSolutionFromBuffers<<<numBlocks, threadsPerBlock>>>(solution_d, evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d);
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	

#ifdef PRINT_SOLUTION
	printf("================NUMBER OF ITERATIONS PERFORMED IN SHARED==============\n");
	printDeviceArrayInt(iterationLevel_d, Ndofs);
	printf("\n================SIMILARITY TO GLOBAL==============\n");
	printDeviceSimilarity1D(solution_d, solutionGM, 1e-6, Ndofs);
	printGlobalAndSharedMatchDevice(solution_d, solutionGM, iterationLevel_d, numIterations, Ndofs);
#endif

#ifdef SAVE_SOLUTION
	// Save iteration level	
	iterationLevelFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/iteration_2.txt");
	for (int i = 0; i < matrix.Ndofs; i++) {
		// printf("Iteration Level[%d] = %d\n", i, iterationLevel[i]);
  		iterationLevelFile << iterationLevel[i];
  		iterationLevelFile << "\n";
	}
  	iterationLevelFile.close();
	// Save territories	
	for (int i = 0; i < bridge.numSubdomains; i++) {
		if (i == 0) {
			territoriesFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/territories_2_1.txt");
		}
		else if (i == 1) {
			territoriesFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/territories_2_2.txt");
		}
		else if (i == 2) {
			territoriesFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/territories_2_3.txt");
		}
		else if (i == 3) {
			territoriesFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/territories_2_4.txt");
		}
		for (auto dof : bridge.territoryDOFsInterior[i]) {
  			territoriesFile << dof;
  			territoriesFile << "\n";
		}
		for (auto dof : bridge.territoryDOFsInteriorExt[i]) {
  			territoriesFile << dof;
  			territoriesFile << "\n";
		}
  		territoriesFile.close();
	}
	uint32_t * iterationLevelBridge = new uint32_t[Ndofs];
	for (int i = 0; i < matrix.Ndofs; i++) {
		iterationLevelBridge[i] = iterationLevel[i];
	} 
#endif
	
	/*************** STAGE 3: LOWER PYRAMIDAL STAGE ************************/
	
	printf("==================== PERFORMING LOWER PYRAMIDAL PARTITIONING =========================\n");

	// INITIALIZE
	meshPartitionForStage lowerPyramidal;
	numExpansionSteps += 5;
	numJacobiSteps += 5;

	// HOST
	set<int> seed3_1 = {149, 165, 173};
	set<int> seed3_2 = {200, 219, 222, 236};
	lowerPyramidal.seeds.push_back(seed3_1);
	lowerPyramidal.seeds.push_back(seed3_2);
	lowerPyramidal.numSubdomains = 2;

	// HOST
	seedsExpandIntoSubdomainsLowerPyramidal(lowerPyramidal, matrix, iterationLevel, numExpansionSteps);
	createHaloRegions(lowerPyramidal, matrix);
	createTerritoriesHost(lowerPyramidal);
	determineMaximumIterationsPerDOF(lowerPyramidal, matrix, iterationLevel, numJacobiSteps);
	constructLocalMatricesHost(lowerPyramidal, matrix);

	// DEVICE
	meshPartitionForStageDevice lowerPyramidal_d;
	allocatePartitionDevice(lowerPyramidal_d, lowerPyramidal, Ndofs);
	copyPartitionDevice(lowerPyramidal_d, lowerPyramidal, Ndofs);
	
	// JACOBI
	determineSharedMemoryAllocation(lowerPyramidal);
	stageAdvanceJacobiPerformance<<<lowerPyramidal.numSubdomains, 512, lowerPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, lowerPyramidal_d, 0, numJacobiSteps); 
		

	// POSTPROCESSING	
	assembleSolutionFromBuffers<<<numBlocks, threadsPerBlock>>>(solution_d, evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d);
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	

#ifdef PRINT_SOLUTION
	printf("================NUMBER OF ITERATIONS PERFORMED IN SHARED==============\n");
	printDeviceArrayInt(iterationLevel_d, Ndofs);
	printf("\n================SIMILARITY TO GLOBAL==============\n");
	printDeviceSimilarity1D(solution_d, solutionGM, 1e-6, Ndofs);
	printGlobalAndSharedMatchDevice(solution_d, solutionGM, iterationLevel_d, numIterations, Ndofs);
#endif

#ifdef SAVE_SOLUTION
	// Save iteration level	
	iterationLevelFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/iteration_3.txt");
	for (int i = 0; i < matrix.Ndofs; i++) {
		// printf("Iteration Level[%d] = %d\n", i, iterationLevel[i]);
  		iterationLevelFile << iterationLevel[i];
  		iterationLevelFile << "\n";
	}
  	iterationLevelFile.close();
	// Return files with the dofs associated with each subdomain	
	for (int i = 0; i < lowerPyramidal.numSubdomains; i++) {
		if (i == 0) {
			territoriesFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/territories_3_1.txt");
		}
		else if (i == 1) {
			territoriesFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/territories_3_2.txt");
		}
		for (auto dof : lowerPyramidal.territoryDOFsInterior[i]) {
  			territoriesFile << dof;
  			territoriesFile << "\n";
		}
		for (auto dof : lowerPyramidal.territoryDOFsInteriorExt[i]) {
  			territoriesFile << dof;
  			territoriesFile << "\n";
		}
  		territoriesFile.close();
	}
#endif
	
	/*************** STAGE 4: DUAL BRIDGE STAGE ************************/
	
	printf("==================== PERFORMING DUAL BRIDGE PARTITIONING =========================\n");
	
	// INITIALIZE
	meshPartitionForStage dualBridge;
	// numExpansionSteps = 4;

	// SEEDS
	set<int> seed4_1;
	int dualBridgeSeed;
	int numSeeds = 174;
	std::ifstream dualBridgeSeedFile1("Unstructured_Mesh/Square_Mesh/Seeds/dualBridgeSeed_1.txt");	
	// Seed 1	
	for (int i = 0; i < numSeeds; i++) {
		dualBridgeSeedFile1 >> dualBridgeSeed;
		seed4_1.insert(dualBridgeSeed);
	}
	dualBridge.seeds.push_back(seed4_1);
	dualBridge.numSubdomains = 1;

	// HOST
	seedsExpandIntoSubdomains(dualBridge, matrix, iterationLevel, numExpansionSteps);
	createHaloRegions(dualBridge, matrix);
	createTerritoriesHost(dualBridge);
	determineMaximumIterationsPerDOF(dualBridge, matrix, iterationLevel, numJacobiSteps);
	constructLocalMatricesHost(dualBridge, matrix);
	
	// DEVICE
	meshPartitionForStageDevice dualBridge_d;
	allocatePartitionDevice(dualBridge_d, dualBridge, Ndofs);
	copyPartitionDevice(dualBridge_d, dualBridge, Ndofs);
	
	// JACOBI
	// Perform Jacobi Iteration kernel call
	determineSharedMemoryAllocation(dualBridge);
	stageAdvanceJacobiPerformance<<<dualBridge.numSubdomains, 512, dualBridge.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, dualBridge_d, 0, numJacobiSteps); 

	// POSTPROCESSING	
	assembleSolutionFromBuffers<<<numBlocks, threadsPerBlock>>>(solution_d, evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d);
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	

#ifdef PRINT_SOLUTION
	printf("================NUMBER OF ITERATIONS PERFORMED IN SHARED==============\n");
	printDeviceArrayInt(iterationLevel_d, Ndofs);
	printf("\n================SIMILARITY TO GLOBAL==============\n");
	printDeviceSimilarity1D(solution_d, solutionGM, 1e-6, Ndofs);
	printGlobalAndSharedMatchDevice(solution_d, solutionGM, iterationLevel_d, numIterations, Ndofs);
#endif

#ifdef SAVE_SOLUTION
	// Save iteration level	
	iterationLevelFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/iteration_4.txt");
	for (int i = 0; i < matrix.Ndofs; i++) {
		// printf("Iteration Level[%d] = %d\n", i, iterationLevel[i]);
  		iterationLevelFile << iterationLevel[i];
  		iterationLevelFile << "\n";
	}
  	iterationLevelFile.close();
	// Return files with the dofs associated with each subdomain	
	for (int i = 0; i < dualBridge.numSubdomains; i++) {
		if (i == 0) {
			territoriesFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/territories_4_1.txt");
		}
		for (auto dof : dualBridge.territoryDOFsInterior[i]) {
  			territoriesFile << dof;
  			territoriesFile << "\n";
		}
		for (auto dof : dualBridge.territoryDOFsInteriorExt[i]) {
  			territoriesFile << dof;
  			territoriesFile << "\n";
		}
  		territoriesFile.close();
	}
#endif
	
	/*************** SUBDOMAIN CONSTRUCTION COMPLETE - PERFORM ACTUAL ITERATIONS ON GPU ************************/
	// Initialize containers for solution and iteration variables
	threadsPerBlock = 128;
	numBlocks = ceil((float)matrix.Ndofs / threadsPerBlock);
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(evenSolutionBuffer_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(oddSolutionBuffer_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(solution_d, matrix.Ndofs);	
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
			maxJacobiShift += 10;
		}

		// Set number of Jacobi iterations for first two stages
		maxJacobiIters += 5;

		// STAGE 1: UPPER PYRAMIDAL
		cudaEventRecord(start_1, 0);
		stageAdvanceJacobiPerformance<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, upperPyramidal_d, minJacobiIters, maxJacobiIters, maxJacobiShift);
		cudaEventRecord(stop_1, 0);
		cudaEventSynchronize(stop_1);
		cudaEventElapsedTime(&time_stage_1, start_1, stop_1);
		time_total_1 += time_stage_1;

		if (sweptIteration > 0) {
			minJacobiIters += 5;
		}
	
		// STAGE 2: BRIDGE STAGE
		cudaEventRecord(start_2, 0);
		stageAdvanceJacobiPerformance<<<bridge.numSubdomains, 512, bridge.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, bridge_d, minJacobiIters, maxJacobiIters, maxJacobiShift);
		cudaEventRecord(stop_2, 0);
		cudaEventSynchronize(stop_2);
		cudaEventElapsedTime(&time_stage_2, start_2, stop_2);
		time_total_2 += time_stage_2;

		// Set number of Jacobi iterations for second two stages
		maxJacobiIters += 5;

		// STAGE 3: LOWER PYRAMIDAL
		cudaEventRecord(start_3, 0);
		stageAdvanceJacobiPerformance<<<lowerPyramidal.numSubdomains, 512, lowerPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, lowerPyramidal_d, minJacobiIters, maxJacobiIters, maxJacobiShift);
		cudaEventRecord(stop_3, 0);
		cudaEventSynchronize(stop_3);
		cudaEventElapsedTime(&time_stage_3, start_3, stop_3);
		time_total_3 += time_stage_3;
		
		// Set number of Jacobi iterations for second two stages
		minJacobiIters += 5;
		
		// Dual Bridge
		cudaEventRecord(start_4, 0);
		stageAdvanceJacobiPerformance<<<dualBridge.numSubdomains, 512, dualBridge.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, dualBridge_d, minJacobiIters, maxJacobiIters, maxJacobiShift);
		cudaEventRecord(stop_4, 0);
		cudaEventSynchronize(stop_4);
		cudaEventElapsedTime(&time_stage_4, start_4, stop_4);
		time_total_4 += time_stage_4;
	
	}

	// Set number of Jacobi iterations for final fill-in stage
	maxJacobiShift += 5;
	bool finalStage = true; 
	// FINAL STAGE
	cudaEventRecord(start_5, 0);
	stageAdvanceJacobiPerformance<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d, upperPyramidal_d, minJacobiIters, maxJacobiIters, maxJacobiShift, finalStage);
	cudaEventRecord(stop_5, 0);
	cudaEventSynchronize(stop_5);
	cudaEventElapsedTime(&time_total_5, start_5, stop_5);

#ifdef SAVE_SOLUTION	
	// Print iteration level
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	
	iterationLevelFile.open("Unstructured_Mesh/Square_Mesh/Iteration_Output/iteration_final.txt");
	for (int i = 0; i < matrix.Ndofs; i++) {
  		iterationLevelFile << iterationLevel[i];
  		iterationLevelFile << "\n";
	}
  	iterationLevelFile.close();
#endif

	// Print information
	printf("\n==================== FINAL INFORMATION =========================\n");
	assembleSolutionFromBuffers<<<numBlocks, threadsPerBlock>>>(solution_d, evenSolutionBuffer_d, oddSolutionBuffer_d, iterationLevel_d);
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

}

