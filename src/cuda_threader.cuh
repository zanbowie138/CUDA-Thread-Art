#pragma once

#include <cassert>
#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include "config.h"

// Tricking intellisense into not being annoying
#ifdef __CUDACC__
#define cuda_SYNCTHREADS() __syncthreads()
#else
#define cuda_SYNCTHREADS()
unsigned int min(unsigned int a, unsigned int b);
#endif

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/*
 * CUDA Core Number: 3072
 * Streaming Multiprocessor (SM) Number: 24
 * Maximum number of threads per block: 1024
 * Maximum number of resident grids per device (Concurrent Kernel Execution): 128
 */

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

typedef union {
	// floats[0] is the value, ints[1] is the index
	float floats[2];
	unsigned int ints[2];
	unsigned long long int ulong;
} SortUnion;



__device__ double calculateRMSError(unsigned char* threadImg, const unsigned char* truthImg, unsigned int imgSize,
                                    unsigned int* lineArray, unsigned int lineStart, unsigned int lineSize);

__global__ void evaluateLineKernel(unsigned char* threadImg, const unsigned char* truthImg,
                                   const unsigned short* lineArray, const unsigned int* lineEndingIdx,
                                   unsigned int globalLineAmt, unsigned int imgSize);

void runCUDAThreader();
