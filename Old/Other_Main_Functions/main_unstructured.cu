using namespace std; 
#include "inttypes.h"
#include <vector>
#include <set>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "structs.h"
#include "helper.h"
#include "mesh.h"
#include "seeds_largeSubdomains.h"
#include "subdomains_largeSubdomains.h"
#include "toDevice.h"
#include "global.h"
#include "swept_update_jacobi_testing.h"
#include "residual.h"
#include "initialize.h"
#include "matrix.h"
#include "print.h"
#include "debug.h"

int main (int argc, char * argv[]) 
{
	// Define number of dofs for the mesh
	uint32_t Ndofs = 103; // this is for the coarse airfoil mesh condensed
	
	// Define the linear system Ax = b

	// Initialize the linear system and allocate matrix data structures (indexPtr, nodeNeighbors, offdiags) 
	linearSystem matrix;
	matrix.Ndofs = Ndofs;
	initializeAndLoadMatrixFromCSRFiles(matrix);   

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
	// uint32_t numIterations = ((nPerSub-1)/2)*2 * numSweptCycles; // 2 * (nPerSub-1) * numSweptCycles;
	uint32_t numIterations = 2; // 2 * (nPerSub-1) * numSweptCycles;
	//uint32_t numJacobiStepsIncrement2 = (((nPerSub-1)/2)*2+1) - ((nPerSub-1)/2 + 1); // nPerSub - ((nPerSub-1)/2 + 1);

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

	// Print solution
	if (numIterations % 2 == 0) { 
		// printDeviceSolution(du0_d, Ndofs);
		solutionGM = du0_d;
	}
	else if (numIterations % 2 == 1) { 
		// printDeviceSolution(du1_d, Ndofs);
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
	printf("GLOBAL MEMORY ITERATIONS: The initial residual is %f but the final residual after %d iterations is %f\n", residualInit, numIterations, residual);
	residualGM = residual;
	
	// SHARED MEMORY START 
	
	printf("==================== SHARED MEMORY ALGORITHM =========================\n");
	
	// Initialize iteration level
	uint32_t * iterationLevel, * iterationLevel_d;
	iterationLevel = new uint32_t[Ndofs];
	initializeToZerosInt(iterationLevel, Ndofs);
	// Iteration Level
    cudaMalloc(&iterationLevel_d, sizeof(uint32_t) * Ndofs);
	cudaMemcpy(iterationLevel_d, iterationLevel, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
	
	// Create swept solution variables on the device
	float * bottomSweptSolution_d, * topSweptSolution_d;
	cudaMalloc(&topSweptSolution_d, sizeof(float) * matrix.Ndofs);
	cudaMalloc(&bottomSweptSolution_d, sizeof(float) * matrix.Ndofs);
	uint32_t threadsPerBlock = 256;
	uint32_t numBlocks = ceil((float)matrix.Ndofs / threadsPerBlock);
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(topSweptSolution_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(bottomSweptSolution_d, matrix.Ndofs);	

	// Number of expansion steps
	// uint32_t numExpansionSteps = nPerSub-1; 

	// Number of Jacobi steps
	uint32_t numJacobiSteps = 3;
	// uint32_t numJacobiStepsIncrement = (nPerSub-1)/2+1;
	
	/*************** STAGE 1: UPPER PYRAMIDAL STAGE ************************/
	
	printf("==================== PERFORMING UPPER PYRAMIDAL PARTITIONING =========================\n");
	
	// HOST
	// Create the partition for this stage (constructing seeds, territories, and local matrix data structures)
	meshPartitionForStage upperPyramidal;
	// Fill in the number of subdomains in this partition, and the seeds themselves
	// constructSeedsHost(upperPyramidal, N, nSub, 0); 
	upperPyramidal.numSubdomains = 4;
	set<int> seed1 = {40};
	set<int> seed2 = {0};
	set<int> seed3 = {100};
	set<int> seed4 = {68};
	upperPyramidal.seeds.push_back(seed1);
	upperPyramidal.seeds.push_back(seed2);
	upperPyramidal.seeds.push_back(seed3);
	upperPyramidal.seeds.push_back(seed4);
	// Create the territories associated with the seeds
	// constructTerritoriesHost(upperPyramidal, matrix, iterationLevel, N, nSub, 0);
	// Determine number of territory expansion steps
	uint32_t numExpansionSteps = 3;
	seedsExpandIntoSubdomains(upperPyramidal, matrix, iterationLevel, numExpansionSteps);
	createHaloRegions(upperPyramidal, matrix);
	createTerritoriesHost(upperPyramidal);
	constructLocalMatricesHost(upperPyramidal, matrix);

	// DEVICE
	// Allocate and copy all partition related data to the GPU
	meshPartitionForStageDevice upperPyramidal_d;
	allocatePartitionDevice(upperPyramidal_d, upperPyramidal, Ndofs);
	copyPartitionDevice(upperPyramidal_d, upperPyramidal, Ndofs);

	// JACOBI
	// Perform Jacobi Iteration kernel call
	determineSharedMemoryAllocation(upperPyramidal);
	numJacobiSteps = 3;
	stageAdvanceJacobiPerformance<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(topSweptSolution_d, bottomSweptSolution_d, iterationLevel_d, upperPyramidal_d, 0, numJacobiSteps, numJacobiSteps);

	// POSTPROCESSING	
	// Print solution
	// printPartitionInformation(upperPyramidal, matrix, topSweptSolution_d, iterationLevel_d, N);	
	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	
	printDeviceArrayInt(iterationLevel_d, Ndofs);
	printf("\n=====================================================\n");
	printDeviceSimilarity1D(topSweptSolution_d, solutionGM, 1e-6, Ndofs);
	printf("\n");
	determineIfValidIterationLevel(iterationLevel, matrix);
	printf("\n");
	
	// Spit out iteration level for visualization in Python
  	ofstream iterationLevelFile;
  	ofstream territoriesFile;
	iterationLevelFile.open("iteration_1.txt");
	territoriesFile.open("territories_1.txt");
	for (int i = 0; i < matrix.Ndofs; i++) {
		// printf("Iteration Level[%d] = %d\n", i, iterationLevel[i]);
  		iterationLevelFile << iterationLevel[i];
  		iterationLevelFile << "\n";
		if (upperPyramidal.subdomainOfDOFs[i] == UINT32_MAX) {
  			territoriesFile << 100;
		}
		else {
  			territoriesFile << upperPyramidal.subdomainOfDOFs[i];
		}
		territoriesFile << "\n";
	}
  	iterationLevelFile.close();
  	territoriesFile.close();
	
	/*************** STAGE 2: BRIDGE STAGE ************************/
	
	printf("==================== PERFORMING BRIDGE PARTITIONING =========================\n");
	
	// HOST
	// Create the partition for this stage (constructing seeds, territories, and local matrix data structures)
	meshPartitionForStage bridge;
	// Fill in the number of subdomains in this partition, and the seeds themselves
	// constructSeedsHost(bridge, N, nSub, 1);
	bridge.numSubdomains = 2;
	// set<int> seed1_2 = {24, 53, 60, 61, 63, 65, 67, 69, 72, 73, 74, 76, 80, 82, 87, 89, 95, 96};
	//set<int> seed2_2 = {71, 75, 77, 81, 83, 84, 85, 90, 91, 92, 93, 98, 99, 101, 102};
	//set<int> seed3_2 = {2, 6, 8, 10, 12, 16, 18, 22, 23, 25, 29, 33, 37, 38, 43, 46, 64};
	//set<int> seed4_2 = {1, 3, 4, 5, 7, 9, 14, 17, 20, 28, 36, 39, 45, 47, 52};
	// set<int> seed1_2 = {2, 8, 15, 18, 27, 34, 38, 43, 46, 50, 51, 57};
	// set<int> seed2_2 = {1, 5, 10, 16, 22, 23, 25, 29, 33, 37, 39};
	// set<int> seed3_2 = {3, 24, 63, 65, 67, 69, 72, 73, 74, 80, 87, 89, 96};
	// set<int> seed4_2 = {75, 76, 77, 82, 83, 85, 80, 92, 93, 95, 98, 102};
	// set<int> seed1_2 = {2, 8, 15, 18, 27, 34, 38, 43, 46, 50, 51, 57, 1, 5, 10, 16, 22, 23, 25, 29, 33, 37, 39, 3, 24, 63, 65, 67, 69, 72, 73, 74, 80, 87, 89, 96, 75, 76, 77, 82, 83, 85, 80, 92, 93, 95, 98, 102};
	set<int> seed1_2 = {1, 2, 5, 8, 10, 15, 16, 18, 22, 23, 25, 27, 29, 33, 34, 37, 38, 39, 43, 44, 46, 50, 51, 57};
	set<int> seed2_2 = {3, 24, 63, 65, 67, 69, 72, 73, 74, 75, 76, 77, 80, 82, 83, 85, 87, 89, 90, 92, 93, 95, 96, 98, 102};
	bridge.seeds.push_back(seed1_2);
	bridge.seeds.push_back(seed2_2);
	// bridge.seeds.push_back(seed3_2);
	// bridge.seeds.push_back(seed4_2);
	// constructTerritoriesHost(bridge, matrix, iterationLevel, N, nSub, 1);
	// Determine number of territory expansion steps
	numExpansionSteps = 2;
	seedsExpandIntoSubdomains(bridge, matrix, iterationLevel, numExpansionSteps);
	createHaloRegions(bridge, matrix);
	createTerritoriesHost(bridge);
	constructLocalMatricesHost(bridge, matrix);
	
	// DEVICE
	// Allocate and copy all partition related data to the GPU
	meshPartitionForStageDevice bridge_d;
	allocatePartitionDevice(bridge_d, bridge, Ndofs);
	copyPartitionDevice(bridge_d, bridge, Ndofs);
	
	// JACOBI
	// Perform Jacobi Iteration kernel call
	determineSharedMemoryAllocation(bridge);
	stageAdvanceJacobiPerformance<<<bridge.numSubdomains, 512, bridge.sharedMemorySize>>>(topSweptSolution_d, bottomSweptSolution_d, iterationLevel_d, bridge_d, 0, 2, numJacobiSteps);
	
	// POSTPROCESSING	
	// Print solution
	// printPartitionInformation(bridge, matrix, topSweptSolution_d, iterationLevel_d, N);	
	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	
	printDeviceArrayInt(iterationLevel_d, Ndofs);
	printf("\n=====================================================\n");
	printDeviceSimilarity1D(topSweptSolution_d, solutionGM, 1e-6, Ndofs);
	printf("\n");
	determineIfValidIterationLevel(iterationLevel, matrix);
	printf("\n");
	
	// Spit out iteration level for visualization in Python
  	// ofstream iterationLevelFile;
  	// ofstream territoriesFile;
	iterationLevelFile.open("iteration_2.txt");
	territoriesFile.open("territories_2.txt");
	for (int i = 0; i < matrix.Ndofs; i++) {
		// printf("Iteration Level[%d] = %d\n", i, iterationLevel[i]);
  		iterationLevelFile << iterationLevel[i];
  		iterationLevelFile << "\n";
		if (bridge.subdomainOfDOFs[i] == UINT32_MAX) {
  			territoriesFile << 100;
		}
		else {
  			territoriesFile << bridge.subdomainOfDOFs[i];
		}
		territoriesFile << "\n";
	}
  	iterationLevelFile.close();
  	territoriesFile.close();
	
	/*************** STAGE 3: LOWER PYRAMIDAL STAGE ************************/
/*
	printf("==================== PERFORMING LOWER PYRAMIDAL PARTITIONING =========================\n");
	// HOST
	// Create the partition for this stage (constructing seeds, territories, and local matrix data structures)
	meshPartitionForStage lowerPyramidal;
	// constructSeedsHost(lowerPyramidal, N, nSub, 2);
	lowerPyramidal.numSubdomains = 2;
	set<int> seed1_3 = {21, 32, 48};
	// set<int> seed2_3 = {34, 50, 54};
	set<int> seed3_3 = {62, 70, 86};
	lowerPyramidal.seeds.push_back(seed1_3);
	// lowerPyramidal.seeds.push_back(seed2_3);
	lowerPyramidal.seeds.push_back(seed3_3);
	// constructTerritoriesHost(lowerPyramidal, matrix, iterationLevel, N, nSub, 2);
	numExpansionSteps = 4;
	seedsExpandIntoSubdomains(lowerPyramidal, matrix, iterationLevel, numExpansionSteps);
	createHaloRegions(lowerPyramidal, matrix);
	createTerritoriesHost(lowerPyramidal);
	constructLocalMatricesHost(lowerPyramidal, matrix);

	// DEVICE
	// Allocate and copy all partition related data to the GPU
	meshPartitionForStageDevice lowerPyramidal_d;
	allocatePartitionDevice(lowerPyramidal_d, lowerPyramidal, Ndofs);
	copyPartitionDevice(lowerPyramidal_d, lowerPyramidal, Ndofs);
	
	// JACOBI
	// Perform Jacobi Iteration kernel call
	determineSharedMemoryAllocation(lowerPyramidal, false);
	numJacobiSteps = 6; // numJacobiStepsIncrement-1;
	stageAdvanceJacobiPerformance<<<lowerPyramidal.numSubdomains, 512, lowerPyramidal.sharedMemorySize>>>(topSweptSolution_d, bottomSweptSolution_d, iterationLevel_d, lowerPyramidal_d, 0, numJacobiSteps, numJacobiSteps);

	// POSTPROCESSING	
	// Print solution
	// printPartitionInformation(lowerPyramidal, matrix, topSweptSolution_d, iterationLevel_d, N);	
	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	
	printDeviceArrayInt(iterationLevel_d, Ndofs);
	printf("\n=====================================================\n");
	printDeviceSimilarity1D(topSweptSolution_d, solutionGM, 1e-6, Ndofs);
	printf("\n");
	determineIfValidIterationLevel(iterationLevel, matrix);
	printf("\n");

	// Spit out iteration level for visualization in Python
  	// ofstream iterationLevelFile;
  	// ofstream territoriesFile;
	iterationLevelFile.open("iteration_3.txt");
	territoriesFile.open("territories_3.txt");
	for (int i = 0; i < matrix.Ndofs; i++) {
		// printf("Iteration Level[%d] = %d\n", i, iterationLevel[i]);
  		iterationLevelFile << iterationLevel[i];
  		iterationLevelFile << "\n";
		if (lowerPyramidal.subdomainOfDOFs[i] == UINT32_MAX) {
  			territoriesFile << 100;
		}
		else {
  			territoriesFile << lowerPyramidal.subdomainOfDOFs[i];
		}
		territoriesFile << "\n";
	}
  	iterationLevelFile.close();
  	territoriesFile.close();
*/	
	/*************** STAGE 4: DUAL BRIDGE STAGE ************************/
/*	
	printf("==================== PERFORMING DUAL BRIDGE PARTITIONING =========================\n");

	// HOST
	// Create the partition for this stage (constructing seeds, territories, and local matrix data structures)
	meshPartitionForStage dualBridge;
	// constructSeedsHost(dualBridge, N, nSub, 3);
	dualBridge.numSubdomains = 1;
	// set<int> seed1_4 = {3, 11, 14, 24, 31, 53, 55, 60, 61, 63, 65, 67, 73};
	// set<int> seed2_4 = {18, 25, 27, 30, 35, 41, 44, 46, 51, 57, 58, 78};
	// set<int> seed3_4 = {69, 72, 76, 80, 82, 83, 84, 85, 88, 90, 91, 94, 97};
	// set<int> seed4_4 = {13, 16, 17, 19, 20, 22, 26, 42, 47, 49, 52, 56};
	set<int> seed1_4 = {3, 11, 14, 24, 31, 53, 55, 60, 61, 63, 65, 67, 73, 18, 25, 27, 30, 35, 41, 44, 46, 50, 51, 57, 58, 78, 69, 72, 76, 80, 82, 83, 84, 85, 88, 90, 91, 94, 97, 13, 16, 17, 19, 20, 22, 26, 42, 47, 49, 52, 56};
	dualBridge.seeds.push_back(seed1_4);
	// dualBridge.seeds.push_back(seed2_4);
	// dualBridge.seeds.push_back(seed3_4);
	// dualBridge.seeds.push_back(seed4_4);
	// constructTerritoriesHost(dualBridge, matrix, iterationLevel, N, nSub, 3);
	numExpansionSteps = 6;
	seedsExpandIntoSubdomains(dualBridge, matrix, iterationLevel, numExpansionSteps);
	createHaloRegions(dualBridge, matrix);
	createTerritoriesHost(dualBridge);
	constructLocalMatricesHost(dualBridge, matrix);
	
	// DEVICE
	// Allocate and copy all partition related data to the GPU
	meshPartitionForStageDevice dualBridge_d;
	allocatePartitionDevice(dualBridge_d, dualBridge, Ndofs);
	copyPartitionDevice(dualBridge_d, dualBridge, Ndofs);
	
	// JACOBI
	// Perform Jacobi Iteration kernel call
	determineSharedMemoryAllocation(dualBridge);
	numJacobiSteps = 6; // numJacobiStepsIncrement-1;
	stageAdvanceJacobiPerformance<<<dualBridge.numSubdomains, 512, dualBridge.sharedMemorySize>>>(topSweptSolution_d, bottomSweptSolution_d, iterationLevel_d, dualBridge_d, 0, 6, numJacobiSteps);

	// POSTPROCESSING	
	// Print solution
	// printPartitionInformation(dualBridge, matrix, topSweptSolution_d, iterationLevel_d, N);	
	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevel_d, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);	
	printDeviceArrayInt(iterationLevel_d, Ndofs);
	printf("\n=====================================================\n");
	printDeviceSimilarity1D(topSweptSolution_d, solutionGM, 1e-6, Ndofs);
	printf("\n");
	determineIfValidIterationLevel(iterationLevel, matrix);
	printf("\n");

	// Spit out iteration level for visualization in Python
  	// ofstream iterationLevelFile;
  	// ofstream territoriesFile;
	iterationLevelFile.open("iteration_4.txt");
	territoriesFile.open("territories_4.txt");
	for (int i = 0; i < matrix.Ndofs; i++) {
		// printf("Iteration Level[%d] = %d\n", i, iterationLevel[i]);
  		iterationLevelFile << iterationLevel[i];
  		iterationLevelFile << "\n";
		if (dualBridge.subdomainOfDOFs[i] == UINT32_MAX) {
  			territoriesFile << 100;
		}
		else {
  			territoriesFile << dualBridge.subdomainOfDOFs[i];
		}
		territoriesFile << "\n";
	}
  	iterationLevelFile.close();
  	territoriesFile.close();
	printDeviceArrayInt(dualBridge_d.territoryDOFs_d, Ndofs);
	
*/	
	/*************** SUBDOMAIN CONSTRUCTION COMPLETE - PERFORM ACTUAL ITERATIONS ON GPU ************************/
/*	
	// Repeat all steps together 
	threadsPerBlock = 128;
	numBlocks = ceil((float)matrix.Ndofs / threadsPerBlock);
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(topSweptSolution_d, matrix.Ndofs);	
	initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(bottomSweptSolution_d, matrix.Ndofs);	
    initializeToZerosInt(iterationLevel, matrix.Ndofs);
    cudaMemcpy(iterationLevel_d, iterationLevel, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
	numJacobiSteps = 0;

    initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(topSweptSolution_d, matrix.Ndofs);
    initializeToZerosDevice<<<numBlocks, threadsPerBlock>>>(bottomSweptSolution_d, matrix.Ndofs);
    initializeToZerosDeviceInt<<<numBlocks, threadsPerBlock>>>(iterationLevel_d, matrix.Ndofs);
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
	uint32_t numSweptCycles = 1;

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
		// numJacobiSteps += numJacobiStepsIncrement-1;
		numJacobiSteps = 3;

		// Stage 1 - Upper Pyramidal
		printf("UPPER PYRAMIDAL STAGE\n");
		cudaEventRecord(start_1, 0);
		stageAdvanceJacobiPerformance<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(topSweptSolution_d, bottomSweptSolution_d, iterationLevel_d, upperPyramidal_d, 0, numJacobiSteps, numJacobiSteps);
		/// printDeviceArrayInt(iterationLevel_d, Ndofs);
		cudaDeviceSynchronize();
		cudaEventRecord(stop_1, 0);
		cudaEventSynchronize(stop_1);
		cudaEventElapsedTime(&time_stage_1, start_1, stop_1);
		time_total_1 += time_stage_1;
		printf("================================================\n");
		printDeviceSimilarity1D(topSweptSolution_d, solutionGM, 1e-6, Ndofs);
		// printPartitionInformation(upperPyramidal, matrix, topSweptSolution_d, iterationLevel_d, N);	
	
		// Stage 2 - Bridge Stage
		printf("\nBRIDGE STAGE\n");
		cudaEventRecord(start_2, 0);
		stageAdvanceJacobiPerformance<<<bridge.numSubdomains, 512, bridge.sharedMemorySize>>>(topSweptSolution_d, bottomSweptSolution_d, iterationLevel_d, bridge_d, 0, numJacobiSteps, numJacobiSteps);
		// printDeviceArrayInt(iterationLevel_d, Ndofs);
		// cudaDeviceSynchronize();
		cudaEventRecord(stop_2, 0);
		cudaEventSynchronize(stop_2);
		cudaEventElapsedTime(&time_stage_2, start_2, stop_2);
		time_total_2 += time_stage_2;
		printf("================================================\n");
		 printDeviceSimilarity1D(topSweptSolution_d, solutionGM, 1e-1, Ndofs);
		// printDeviceSolutionComparison(topSweptSolution_d, solutionGM, Ndofs);
		// printPartitionInformation(upperPyramidal, matrix, topSweptSolution_d, iterationLevel_d, N);	

		// Set number of Jacobi iterations for second two stages
		// numJacobiSteps += numJacobiStepsIncrement-1;
		numJacobiSteps = 6;

		// Lower Pyramidal
		printf("\nLOWER PYRAMIDAL STAGE\n");
		cudaEventRecord(start_3, 0);
		stageAdvanceJacobiPerformance<<<lowerPyramidal.numSubdomains, 512, lowerPyramidal.sharedMemorySize>>>(topSweptSolution_d, bottomSweptSolution_d, iterationLevel_d, lowerPyramidal_d, 0, numJacobiSteps, numJacobiSteps);
		// printDeviceArrayInt(iterationLevel_d, Ndofs);
		// cudaDeviceSynchronize();
		cudaEventRecord(stop_3, 0);
		cudaEventSynchronize(stop_3);
		cudaEventElapsedTime(&time_stage_3, start_3, stop_3);
		time_total_3 += time_stage_3;
		printf("================================================\n");
		printDeviceSimilarity1D(topSweptSolution_d, solutionGM, 1e-6, Ndofs);
		// printPartitionInformation(upperPyramidal, matrix, topSweptSolution_d, iterationLevel_d, N);	

		// Dual Bridge
		printf("\nDUAL BRIDGE STAGE\n");
		cudaEventRecord(start_4, 0);
		stageAdvanceJacobiPerformance<<<dualBridge.numSubdomains, 512, dualBridge.sharedMemorySize>>>(topSweptSolution_d, bottomSweptSolution_d, iterationLevel_d, dualBridge_d, 0, numJacobiSteps, numJacobiSteps);
		// printDeviceArrayInt(iterationLevel_d, Ndofs);
		// cudaDeviceSynchronize();
		cudaEventRecord(stop_4, 0);
		cudaEventSynchronize(stop_4);
		cudaEventElapsedTime(&time_stage_4, start_4, stop_4);
		time_total_4 += time_stage_4;
		printf("================================================\n");
		printDeviceSimilarity1D(topSweptSolution_d, solutionGM, 1e-6, Ndofs);
		// printPartitionInformation(upperPyramidal, matrix, topSweptSolution_d, iterationLevel_d, N);	
	}

	// Set number of Jacobi iterations for final fill-in stage
	// numJacobiSteps += numJacobiStepsIncrement;
	// Final Stage	
	printf("\nFINAL STAGE\n");
	// JACOBI
	cudaEventRecord(start_5, 0);
	// stageAdvanceJacobiPerformance<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(topSweptSolution_d, bottomSweptSolution_d, iterationLevel_d, upperPyramidal_d, 2*(nPerSub-1), 3*(nPerSub-1));
	bool finalStage = true;
	stageAdvanceJacobiPerformance<<<upperPyramidal.numSubdomains, 512, upperPyramidal.sharedMemorySize>>>(topSweptSolution_d, bottomSweptSolution_d, iterationLevel_d, upperPyramidal_d, 0, 6, 9, finalStage);
	printDeviceArrayInt(iterationLevel_d, Ndofs);
	// printf("================================================\n");
	printDeviceSimilarity1D(topSweptSolution_d, solutionGM, 1e-6, Ndofs);
	// cudaDeviceSynchronize();
	cudaEventRecord(stop_5, 0);
	cudaEventSynchronize(stop_5);
	cudaEventElapsedTime(&time_total_5, start_5, stop_5);

	// Print information
	
	printf("\n==================== FINAL INFORMATION =========================\n");
	
	// Print solution for swept algorithm

	printf("SOLUTION\n");
	printf("GLOBAL MEMORY SOLUTION AFTER %d ITERATIONS\n", numIterations);
	// printDeviceSolutionFloat(solutionGM, Ndofs, N);
	printf("SHARED MEMORY (SWEPT) SOLUTION AFTER %d ITERATIONS\n", numIterations);

	// Compute L2 residual
	residual = computeL2Residual(topSweptSolution_d, matrix_d);
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

