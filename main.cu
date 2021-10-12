using namespace std; 
#include "inttypes.h"
#include "global.h"
#include <vector>
#include <set>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "helper.h"
#include "mesh_construction.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "seeds.h"
#include "swept_functions.h"
//#include "bridge.h"
// #include "structs.h"

int main (int argc, char * argv[]) 
{
    // Inputs
    uint32_t numIterationsGlobal = 1;
    uint32_t N = 6;

    // Define solution
	uint32_t Ndofs = N*N;
    float *du0 = new float[Ndofs];
    float *du1 = new float[Ndofs];
    initializeToZeros(du0, Ndofs);
    initializeToZeros(du1, Ndofs);
    float *du0_d;
    float *du1_d;
    cudaMalloc(&du0_d, sizeof(float) * Ndofs);
    cudaMalloc(&du1_d, sizeof(float) * Ndofs);
    cudaMemcpy(du0_d, du0, sizeof(float) * Ndofs, cudaMemcpyHostToDevice);
    cudaMemcpy(du1_d, du1, sizeof(float) * Ndofs, cudaMemcpyHostToDevice);
 
    // Define rhs
    float *rhs = new float[Ndofs];
    initializeToOnes(rhs, Ndofs);
    float *rhs_d;
    cudaMalloc(&rhs_d, sizeof(float) * Ndofs);
    cudaMemcpy(rhs_d, rhs, sizeof(float) * Ndofs, cudaMemcpyHostToDevice);

    // Perform iterations in global memory
    // globalMemorySolve2D(du1_d, du0_d, N, numIterations); 

	/* SHARED MEMORY ALGORITHM */ 
 
    // Create the idxPtr and nodeNeighbors for 2D Structured
    uint32_t numEntries = (N-1)*(N-1)*8 + 4*(N-2)*5 + 4*3;

	// Allocate matrix data on the CPU
    uint32_t * indexPtr = new uint32_t[Ndofs+1];
    uint32_t * nodeNeighbors = new uint32_t[numEntries];
    construct2DConnectivity_DiagonalLinks(indexPtr, nodeNeighbors, N);

    // Allocate matrix data structures to the GPU 
    uint32_t * indexPtrGPU, * nodeNeighborsGPU;
    cudaMalloc(&indexPtrGPU, sizeof(uint32_t) * (Ndofs+1));
    cudaMalloc(&nodeNeighborsGPU, sizeof(uint32_t) * numEntries);
    cudaMemcpy(indexPtrGPU, indexPtr, sizeof(uint32_t) * (Ndofs+1), cudaMemcpyHostToDevice);
    cudaMemcpy(nodeNeighborsGPU, nodeNeighbors, sizeof(uint32_t) * numEntries, cudaMemcpyHostToDevice);


	/*************** UPPER PYRAMIDAL STAGE ************************/
	// Initialize iteration level
	uint32_t * iterationLevel = new uint32_t[Ndofs];
	initializeToZerosInt(iterationLevel, Ndofs);

    // Create initial seeds vector
    uint32_t numSubdomains = 4;
	vector<set<int>> seedsUpperPyramidal;
	constructSeedsUpperPyramidal(seedsUpperPyramidal, numSubdomains, N);
	for (auto setOfDofs : seedsUpperPyramidal) {
		for (auto dof : setOfDofs) {
			printf("dof = %d\n", dof);
		}
	}
	
	// Initialize seed distance to help determine when to advance
	uint32_t * distanceFromSeedUP = new uint32_t[Ndofs];

	// Create territories
	vector<set<int>> territoryUpperPyramidal;
	vector<set<int>> territoryUpperPyramidalExpanded;
	uint32_t numExpansionSteps = N/2-2;
	seedsExpandIntoSubdomains(territoryUpperPyramidal, seedsUpperPyramidal, iterationLevel, distanceFromSeedUP, indexPtr, nodeNeighbors, Ndofs, numSubdomains, numExpansionSteps);
	expandToHaloRegions(territoryUpperPyramidalExpanded, territoryUpperPyramidal, indexPtr, nodeNeighbors, numSubdomains);

	// Create territory data and territory Index Ptr
	int * territoryIndexPtr = new int[numSubdomains+1];
	int * territoryIndexPtrExpanded = new int[numSubdomains+1];
	territoryIndexPtr[0] = 0;
	territoryIndexPtrExpanded[0] = 0;
	for (int i = 0; i < numSubdomains; i++) {
		territoryIndexPtr[i+1] = territoryIndexPtr[i] + territoryUpperPyramidal[i].size();
		territoryIndexPtrExpanded[i+1] = territoryIndexPtrExpanded[i] + territoryUpperPyramidalExpanded[i].size();
	}
	uint32_t numElems = territoryIndexPtr[numSubdomains];
	uint32_t numElemsExpanded = territoryIndexPtrExpanded[numSubdomains];
	uint32_t * territoryDOFs = new uint32_t[numElems];
	uint32_t * territoryDOFsExpanded = new uint32_t[numElemsExpanded];
	uint32_t idx1 = 0;
	uint32_t idx2 = 0;
	for (int i = 0; i < numSubdomains; i++) {
		for (auto elem : territoryUpperPyramidal[i]) {
			territoryDOFs[idx1] = elem;
			// printf("Subdomain %d has element %d\n", i, elem);
			idx1 += 1;
		}
		for (auto elem : territoryUpperPyramidalExpanded[i]) {
			territoryDOFsExpanded[idx2] = elem;
			// printf("Subdomain %d has element %d\n", i, elem);
			idx2 += 1;
		}
	}

	// Allocate memory on the GPU for all non-matrix related data
	// Iteration Level
	uint32_t * iterationLevelGPU;
    cudaMalloc(&iterationLevelGPU, sizeof(uint32_t) * Ndofs);
	cudaMemcpy(iterationLevelGPU, iterationLevel, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
	// Seed Distance
	uint32_t * distanceFromSeedUPGPU;
    cudaMalloc(&distanceFromSeedUPGPU, sizeof(uint32_t) * Ndofs);
	cudaMemcpy(distanceFromSeedUPGPU, distanceFromSeedUP, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
	// Territories
	uint32_t * territoryDOFsGPU, * territoryDOFsExpandedGPU;
	cudaMalloc(&territoryDOFsGPU, sizeof(uint32_t) * numElems);
	cudaMalloc(&territoryDOFsExpandedGPU, sizeof(uint32_t) * numElemsExpanded);
	cudaMemcpy(territoryDOFsGPU, territoryDOFs, sizeof(uint32_t) * numElems, cudaMemcpyHostToDevice);	
	cudaMemcpy(territoryDOFsExpandedGPU, territoryDOFsExpanded, sizeof(uint32_t) * numElemsExpanded, cudaMemcpyHostToDevice);	
	// Territories Index Pointer
	uint32_t * territoryIndexPtrGPU, * territoryIndexPtrExpandedGPU;
	cudaMalloc(&territoryIndexPtrGPU, sizeof(uint32_t) * (numSubdomains+1));
	cudaMalloc(&territoryIndexPtrExpandedGPU, sizeof(uint32_t) * (numSubdomains+1));
	cudaMemcpy(territoryIndexPtrGPU, territoryIndexPtr, sizeof(uint32_t) * (numSubdomains+1), cudaMemcpyHostToDevice);
	cudaMemcpy(territoryIndexPtrExpandedGPU, territoryIndexPtrExpanded, sizeof(uint32_t) * (numSubdomains+1), cudaMemcpyHostToDevice);
	
	// UPPER PYRAMIDAL STAGE KERNEL
	numExpansionSteps = N/2-1;
	upperPyramidalNew(du0_d, territoryDOFsGPU, territoryIndexPtrGPU, territoryDOFsExpandedGPU, territoryIndexPtrExpandedGPU, iterationLevelGPU, distanceFromSeedUPGPU, indexPtrGPU, nodeNeighborsGPU, Ndofs, numSubdomains, numExpansionSteps, numElemsExpanded);
	float * solution = new float[Ndofs];
	cudaMemcpy(solution, du0_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%d ", (uint32_t)solution[dof]);
		}
		printf("\n");
	}	
	printf("===================================\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%d ", (uint32_t)solution[dof]);
		}
		printf("\n");
	}	
	
	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevelGPU, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);

	/*************** BRIDGE STAGE ************************/
	
	// Create initial seeds vector
	vector<set<int>> seedsBridge;
	constructSeedsBridge(seedsBridge, numSubdomains, N);
	for (auto setOfDofs : seedsBridge) {
		printf("===========SEED SET=============\n");
		for (auto dof : setOfDofs) {
			printf("dof = %d\n", dof);
		}
	}
	
	// Initialize seed distance to help determine when to advance
	uint32_t * distanceFromSeedBridge = new uint32_t[Ndofs];

	// Create territories
	vector<set<int>> territoryBridge;
	vector<set<int>> territoryBridgeExpanded;
	numExpansionSteps = N/2-2;
	seedsExpandIntoSubdomains(territoryBridge, seedsBridge, iterationLevel, distanceFromSeedBridge, indexPtr, nodeNeighbors, Ndofs, numSubdomains, numExpansionSteps);
	expandToHaloRegions(territoryBridgeExpanded, territoryBridge, indexPtr, nodeNeighbors, numSubdomains);
	for (int i = 0; i < numSubdomains; i++) {
		for (auto elem : territoryBridgeExpanded[i]) {
			printf("Subdomain %d has member %d\n", i, elem);
		}
	}

	// Create territory data and territory Index Ptr
	int * territoryIndexPtrBridge = new int[numSubdomains+1];
	int * territoryIndexPtrBridgeExpanded = new int[numSubdomains+1];
	territoryIndexPtrBridge[0] = 0;
	territoryIndexPtrBridgeExpanded[0] = 0;
	for (int i = 0; i < numSubdomains; i++) {
		territoryIndexPtrBridge[i+1] = territoryIndexPtrBridge[i] + territoryBridge[i].size();
		territoryIndexPtrBridgeExpanded[i+1] = territoryIndexPtrBridgeExpanded[i] + territoryBridgeExpanded[i].size();
	}
	numElems = territoryIndexPtrBridge[numSubdomains];
	numElemsExpanded = territoryIndexPtrBridgeExpanded[numSubdomains];
	uint32_t * territoryDOFsBridge = new uint32_t[numElems];
	uint32_t * territoryDOFsBridgeExpanded = new uint32_t[numElemsExpanded];
	idx1 = 0;
	idx2 = 0;
	for (int i = 0; i < numSubdomains; i++) {
		for (auto elem : territoryBridge[i]) {
			territoryDOFsBridge[idx1] = elem;
			idx1 += 1;
		}
		for (auto elem : territoryBridgeExpanded[i]) {
			territoryDOFsBridgeExpanded[idx2] = elem;
			idx2 += 1;
		}
	}

	// Allocate memory on the GPU for all non-matrix related data
	// Seed Distance
	uint32_t * distanceFromSeedBridgeGPU;
    cudaMalloc(&distanceFromSeedBridgeGPU, sizeof(uint32_t) * Ndofs);
	cudaMemcpy(distanceFromSeedBridgeGPU, distanceFromSeedBridge, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
	// Territories
	uint32_t * territoryDOFsBridgeGPU, * territoryDOFsBridgeExpandedGPU;
	cudaMalloc(&territoryDOFsBridgeGPU, sizeof(uint32_t) * numElems);
	cudaMalloc(&territoryDOFsBridgeExpandedGPU, sizeof(uint32_t) * numElemsExpanded);
	cudaMemcpy(territoryDOFsBridgeGPU, territoryDOFsBridge, sizeof(uint32_t) * numElems, cudaMemcpyHostToDevice);	
	cudaMemcpy(territoryDOFsBridgeExpandedGPU, territoryDOFsBridgeExpanded, sizeof(uint32_t) * numElemsExpanded, cudaMemcpyHostToDevice);	
	// Territories Index Pointer
	uint32_t * territoryIndexPtrBridgeGPU, * territoryIndexPtrBridgeExpandedGPU;
	cudaMalloc(&territoryIndexPtrBridgeGPU, sizeof(uint32_t) * (numSubdomains+1));
	cudaMalloc(&territoryIndexPtrBridgeExpandedGPU, sizeof(uint32_t) * (numSubdomains+1));
	cudaMemcpy(territoryIndexPtrBridgeGPU, territoryIndexPtrBridge, sizeof(uint32_t) * (numSubdomains+1), cudaMemcpyHostToDevice);
	cudaMemcpy(territoryIndexPtrBridgeExpandedGPU, territoryIndexPtrBridgeExpanded, sizeof(uint32_t) * (numSubdomains+1), cudaMemcpyHostToDevice);

	// BRIDGE STAGE KERNEL
	numExpansionSteps = N/2-1;
	upperPyramidalNew(du0_d, territoryDOFsBridgeGPU, territoryIndexPtrBridgeGPU, territoryDOFsBridgeExpandedGPU, territoryIndexPtrBridgeExpandedGPU, iterationLevelGPU, distanceFromSeedBridgeGPU, indexPtrGPU, nodeNeighborsGPU, Ndofs, numSubdomains, numExpansionSteps, numElemsExpanded);
	cudaMemcpy(solution, du0_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%d ", (uint32_t)solution[dof]);
		}
		printf("\n");
	}	
	printf("===================================\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%d ", (uint32_t)solution[dof]);
		}
		printf("\n");
	}	

	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevelGPU, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);

	/*************** LOWER PYRAMIDAL STAGE ************************/
	
	// Create initial seeds vector
    numSubdomains = 1;
	vector<set<int>> seedsLP;
	constructSeedsLowerPyramidal(seedsLP, numSubdomains, N);
	for (auto setOfDofs : seedsLP) {
		printf("===========SEED SET=============\n");
		for (auto dof : setOfDofs) {
			printf("dof = %d\n", dof);
		}
	}

	// Initialize seed distance to help determine when to advance
	uint32_t * distanceFromSeedLP = new uint32_t[Ndofs];

	// Create territories
	vector<set<int>> territoryLP;
	vector<set<int>> territoryLPExpanded;
	numExpansionSteps = N/2-1;
	seedsExpandIntoSubdomains(territoryLP, seedsLP, iterationLevel, distanceFromSeedLP, indexPtr, nodeNeighbors, Ndofs, numSubdomains, numExpansionSteps);
	expandToHaloRegions(territoryLPExpanded, territoryLP, indexPtr, nodeNeighbors, numSubdomains);
	for (int i = 0; i < numSubdomains; i++) {
		for (auto elem : territoryLPExpanded[i]) {
			printf("Subdomain %d has member %d\n", i, elem);
		}
	}

	// Create territory data and territory index Ptr
	int * territoryIndexPtrLP = new int[numSubdomains+1];
	int * territoryIndexPtrLPExpanded = new int[numSubdomains+1];
	territoryIndexPtrLP[0] = 0;
	territoryIndexPtrLPExpanded[0] = 0;
	for (int i = 0; i < numSubdomains; i++) {
		territoryIndexPtrLP[i+1] = territoryIndexPtrLP[i] + territoryLP[i].size();
		territoryIndexPtrLPExpanded[i+1] = territoryIndexPtrLPExpanded[i] + territoryLPExpanded[i].size();
	}
	numElems = territoryIndexPtrLP[numSubdomains];
	numElemsExpanded = territoryIndexPtrLPExpanded[numSubdomains];
	uint32_t * territoryDOFsLP = new uint32_t[numElems];
	uint32_t * territoryDOFsLPExpanded = new uint32_t[numElemsExpanded];
	idx1 = 0;
	idx2 = 0;
	for (int i = 0; i < numSubdomains; i++) {
		for (auto elem : territoryLP[i]) {
			territoryDOFsLP[idx1] = elem;
			printf("Subdomain %d has element %d at entry %d\n", i, territoryDOFsLP[idx1], idx1);
			idx1 += 1;
		}
		for (auto elem : territoryLPExpanded[i]) {
			territoryDOFsLPExpanded[idx2] = elem;
			printf("Expanded Subdomain %d has element %d at entry %d\n", i, territoryDOFsLPExpanded[idx2], idx2);
			idx2 += 1;
		}
	}

	// Allocate memory on the GPU for all non-matrix related data
	// Seed Distance
	uint32_t * distanceFromSeedLPGPU;
    cudaMalloc(&distanceFromSeedLPGPU, sizeof(uint32_t) * Ndofs);
	cudaMemcpy(distanceFromSeedLPGPU, distanceFromSeedLP, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
	// Territories
	uint32_t * territoryDOFsLPGPU, * territoryDOFsLPExpandedGPU;
	cudaMalloc(&territoryDOFsLPGPU, sizeof(uint32_t) * numElems);
	cudaMalloc(&territoryDOFsLPExpandedGPU, sizeof(uint32_t) * numElemsExpanded);
	cudaMemcpy(territoryDOFsLPGPU, territoryDOFsLP, sizeof(uint32_t) * numElems, cudaMemcpyHostToDevice);	
	cudaMemcpy(territoryDOFsLPExpandedGPU, territoryDOFsLPExpanded, sizeof(uint32_t) * numElemsExpanded, cudaMemcpyHostToDevice);	
	// Territories Index Pointer
	uint32_t * territoryIndexPtrLPGPU, * territoryIndexPtrLPExpandedGPU;
	cudaMalloc(&territoryIndexPtrLPGPU, sizeof(uint32_t) * (numSubdomains+1));
	cudaMalloc(&territoryIndexPtrLPExpandedGPU, sizeof(uint32_t) * (numSubdomains+1));
	cudaMemcpy(territoryIndexPtrLPGPU, territoryIndexPtrLP, sizeof(uint32_t) * (numSubdomains+1), cudaMemcpyHostToDevice);
	cudaMemcpy(territoryIndexPtrLPExpandedGPU, territoryIndexPtrLPExpanded, sizeof(uint32_t) * (numSubdomains+1), cudaMemcpyHostToDevice);

	// LOWER PYRAMIDAL KERNEL
	numExpansionSteps = 4;
	upperPyramidalNew(du0_d, territoryDOFsLPGPU, territoryIndexPtrLPGPU, territoryDOFsLPExpandedGPU, territoryIndexPtrLPExpandedGPU, iterationLevelGPU, distanceFromSeedLPGPU, indexPtrGPU, nodeNeighborsGPU, Ndofs, numSubdomains, numExpansionSteps, numElemsExpanded);
	cudaMemcpy(solution, du0_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%d ", (uint32_t)solution[dof]);
		}
		printf("\n");
	}	
	printf("===================================\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%d ", (uint32_t)solution[dof]);
		}
		printf("\n");
	}	

	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevelGPU, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);

	/*************** DUAL BRIDGE STAGE ************************/
	
	// Create initial seeds vector
    numSubdomains = 4;
	vector<set<int>> seedsDB;
	constructSeedsDualBridge(seedsDB, numSubdomains, N);
	for (auto setOfDofs : seedsLP) {
		printf("===========SEED SET=============\n");
		for (auto dof : setOfDofs) {
			printf("dof = %d\n", dof);
		}
	}

	// Initialize seed distance to help determine when to advance
	uint32_t * distanceFromSeedDB = new uint32_t[Ndofs];

	// Create territories
	vector<set<int>> territoryDB;
	vector<set<int>> territoryDBExpanded;
	numExpansionSteps = N/2;
	seedsExpandIntoSubdomains(territoryDB, seedsDB, iterationLevel, distanceFromSeedDB, indexPtr, nodeNeighbors, Ndofs, numSubdomains, numExpansionSteps);
	expandToHaloRegions(territoryDBExpanded, territoryDB, indexPtr, nodeNeighbors, numSubdomains);
	for (int i = 0; i < numSubdomains; i++) {
		for (auto elem : territoryDBExpanded[i]) {
			printf("Subdomain %d has member %d\n", i, elem);
		}
	}

	// Create territory data and territory index Ptr
	int * territoryIndexPtrDB = new int[numSubdomains+1];
	int * territoryIndexPtrDBExpanded = new int[numSubdomains+1];
	territoryIndexPtrDB[0] = 0;
	territoryIndexPtrDBExpanded[0] = 0;
	for (int i = 0; i < numSubdomains; i++) {
		territoryIndexPtrDB[i+1] = territoryIndexPtrDB[i] + territoryDB[i].size();
		territoryIndexPtrDBExpanded[i+1] = territoryIndexPtrDBExpanded[i] + territoryDBExpanded[i].size();
	}
	numElems = territoryIndexPtrDB[numSubdomains];
	numElemsExpanded = territoryIndexPtrDBExpanded[numSubdomains];
	uint32_t * territoryDOFsDB = new uint32_t[numElems];
	uint32_t * territoryDOFsDBExpanded = new uint32_t[numElemsExpanded];
	idx1 = 0;
	idx2 = 0;
	for (int i = 0; i < numSubdomains; i++) {
		for (auto elem : territoryDB[i]) {
			territoryDOFsDB[idx1] = elem;
			printf("Subdomain %d has element %d at entry %d with seed distance %d\n", i, territoryDOFsDB[idx1], idx1, distanceFromSeedDB[elem]);
			idx1 += 1;
		}
		for (auto elem : territoryDBExpanded[i]) {
			territoryDOFsDBExpanded[idx2] = elem;
			printf("Expanded Subdomain %d has element %d at entry %d\n", i, territoryDOFsDBExpanded[idx2], idx2);
			idx2 += 1;
		}
	}

	// Allocate memory on the GPU for all non-matrix related data
	// Seed Distance
	uint32_t * distanceFromSeedDBGPU;
    cudaMalloc(&distanceFromSeedDBGPU, sizeof(uint32_t) * Ndofs);
	cudaMemcpy(distanceFromSeedDBGPU, distanceFromSeedDB, sizeof(uint32_t) * Ndofs, cudaMemcpyHostToDevice);
	// Territories
	uint32_t * territoryDOFsDBGPU, * territoryDOFsDBExpandedGPU;
	cudaMalloc(&territoryDOFsDBGPU, sizeof(uint32_t) * numElems);
	cudaMalloc(&territoryDOFsDBExpandedGPU, sizeof(uint32_t) * numElemsExpanded);
	cudaMemcpy(territoryDOFsDBGPU, territoryDOFsDB, sizeof(uint32_t) * numElems, cudaMemcpyHostToDevice);	
	cudaMemcpy(territoryDOFsDBExpandedGPU, territoryDOFsDBExpanded, sizeof(uint32_t) * numElemsExpanded, cudaMemcpyHostToDevice);	
	// Territories Index Pointer
	uint32_t * territoryIndexPtrDBGPU, * territoryIndexPtrDBExpandedGPU;
	cudaMalloc(&territoryIndexPtrDBGPU, sizeof(uint32_t) * (numSubdomains+1));
	cudaMalloc(&territoryIndexPtrDBExpandedGPU, sizeof(uint32_t) * (numSubdomains+1));
	cudaMemcpy(territoryIndexPtrDBGPU, territoryIndexPtrDB, sizeof(uint32_t) * (numSubdomains+1), cudaMemcpyHostToDevice);
	cudaMemcpy(territoryIndexPtrDBExpandedGPU, territoryIndexPtrDBExpanded, sizeof(uint32_t) * (numSubdomains+1), cudaMemcpyHostToDevice);

	// DUAL BRIDGE KERNEL
	numExpansionSteps = 4;
	upperPyramidalNew(du0_d, territoryDOFsDBGPU, territoryIndexPtrDBGPU, territoryDOFsDBExpandedGPU, territoryIndexPtrDBExpandedGPU, iterationLevelGPU, distanceFromSeedDBGPU, indexPtrGPU, nodeNeighborsGPU, Ndofs, numSubdomains, numExpansionSteps, numElemsExpanded);
	cudaMemcpy(solution, du0_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%d ", (uint32_t)solution[dof]);
		}
		printf("\n");
	}	
	printf("===================================\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%d ", (uint32_t)solution[dof]);
		}
		printf("\n");
	}	

	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevelGPU, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);

	/*************** REPEAT UPPER PYRAMIDAL STAGE TO COMPLETION ************************/
	
	// UPPER PYRAMIDAL STAGE KERNEL
	numExpansionSteps = 6;
	upperPyramidalNew(du0_d, territoryDOFsGPU, territoryIndexPtrGPU, territoryDOFsExpandedGPU, territoryIndexPtrExpandedGPU, iterationLevelGPU, distanceFromSeedUPGPU, indexPtrGPU, nodeNeighborsGPU, Ndofs, numSubdomains, numExpansionSteps, numElemsExpanded);
	cudaMemcpy(solution, du0_d, sizeof(float) * Ndofs, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%d ", (uint32_t)solution[dof]);
		}
		printf("\n");
	}	
	printf("===================================\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			uint32_t dof = j + i * N;
			printf("%d ", (uint32_t)solution[dof]);
		}
		printf("\n");
	}	
	
	// Copy iterative level back to CPU so partitioning can be done for next step
	cudaMemcpy(iterationLevel, iterationLevelGPU, sizeof(uint32_t) * Ndofs, cudaMemcpyDeviceToHost);

	vector<uint32_t*> allSubdivisions;
	allSubdivisions.push_back(territoryDOFsExpanded);

}
