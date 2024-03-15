#include "cuda_threader.cuh"

#include "utils.h"

__device__ float calculateRMSError(const unsigned char* threadImg, const unsigned char* truthImg,
                                   const unsigned int imgSize,
                                   const unsigned short* lineArray, const unsigned int lineStart,
                                   const unsigned int lineSize)
{
	float avg = 0.0;
	const float count = lineSize / 2;

	if (lineSize == 0)
	{
		printf("Empty RMS\n");
		assert(false && "Empty RMS");
	}

	for (int i = 0; i < lineSize; i += 2)
	{
		unsigned short x = lineArray[lineStart + i];
		unsigned short y = lineArray[lineStart + i + 1];
		float diff = truthImg[y * imgSize + x];
		avg += diff * diff / count;
	}

	return sqrtf(avg);
}

typedef union {
	float floats[2];
	unsigned int ints[2];\
	unsigned long long int ulong;
} SortUnion;

// TODO: Check functionality
// I have no clue how this works...
__device__ unsigned long long int atomicMinIndex(unsigned long long int* array, float value, int idx) {
	SortUnion thisVal, testVal;
	thisVal.floats[0] = value;
	thisVal.ints[1] = idx;
	testVal.ulong = *array;
	while (testVal.floats[0] > value)
	{
		testVal.ulong = atomicCAS(array, testVal.ulong, thisVal.ulong);
	}
	return testVal.ulong;
}


__global__ void evaluateLineKernel(unsigned char* threadImg, unsigned char* truthImg,
                                   const unsigned short* lineArray, const unsigned int* lineEndingIdx,
                                   unsigned int globalLineAmt, unsigned int imgSize)
{
	const unsigned int tIdx = threadIdx.x;
	//printf("Thread %d: Starting...\n", tIdx);
	if (tIdx >= UNIQUE_LINE_NUMBER) { return; }

	const unsigned int linesPerThread = static_cast<unsigned int>(
		floor(static_cast<double>(UNIQUE_LINE_NUMBER) / blockDim.x));

	const unsigned int remainder = UNIQUE_LINE_NUMBER % blockDim.x;
	unsigned int threadLineAmt;
	if (tIdx < remainder)
	{
		threadLineAmt = linesPerThread + 1;
	}
	else
	{
		threadLineAmt = linesPerThread;
	}

	//printf("Thread %d: Lines Amt: %d\n", tIdx, threadLineAmt);

	const auto lineStarts = new unsigned int[threadLineAmt];
	const auto lineSizes = new unsigned int[threadLineAmt];
	auto finishedLines = new bool[threadLineAmt];
	unsigned int finishedLinesAmt = 0;

	for (int i = 0; i < threadLineAmt; i++)
	{
		unsigned int lineIdx;
		if (tIdx < remainder)
		{
			lineIdx = tIdx * (linesPerThread + 1) + i;
		}
		else
		{
			lineIdx = tIdx * linesPerThread + i + remainder;
		}
		//printf("Thread %d: Line %d: Index: %d\n", tIdx, i, lineIdx);

		if (lineIdx != 0)
		{
			lineStarts[i] = lineEndingIdx[lineIdx - 1] * 2 + 1;
			lineSizes[i] = (lineEndingIdx[lineIdx] - lineEndingIdx[lineIdx - 1]) * 2;
		}
		else
		{
			lineStarts[i] = 0;
			lineSizes[i] = lineEndingIdx[0] * 2;
		}
		//printf("Thread %d: Line %d: Start: %d, Size: %d\n", tIdx, lineIdx, lineStarts[i], lineSizes[i]);
	}
	cuda_SYNCTHREADS();

	__shared__ unsigned long long int globalBestRMS;

	for (unsigned int l = 0; l < globalLineAmt; l++)
	{
		// Reset global best
		globalBestRMS = ((unsigned long long int)99999 << 32) | __float_as_uint(99999);
		float bestRMS = 99999;
		unsigned int bestRMSIdx = 9999999;
		unsigned int bestLocalIdx = 9999;
		for (int j = 0; j < threadLineAmt; j += 1)
		{
			float error = calculateRMSError(threadImg, truthImg, imgSize, lineArray, lineStarts[j], lineSizes[j]);
			//printf("Line %d: Error: %f\n", tIdx * linesPerThread + j, error);
			if (error < bestRMS && !finishedLines[j])
			{
				bestRMS = error;
				bestRMSIdx = tIdx;
				bestLocalIdx = j;
				if (error == 0) { break; }
				//printf("Thread %d: Best Intermediate RMS: %f , Best Intermediate RMS Index: %d\n", tIdx, bestRMS, bestLocalIdx);
			}
		}

		cuda_SYNCTHREADS();

		// Find the best RMS from all threads
		atomicMinIndex(&globalBestRMS, bestRMS, bestRMSIdx);

		unsigned int bestThreadIdx = (globalBestRMS >> 32);
		float bestRMSValue = __uint_as_float(globalBestRMS & 0xFFFFFFFF);

		if (bestRMSValue > bestRMS)
		{
			printf("Error: Best RMS: %f, Global Best RMS: %f\n", bestRMS, bestRMSValue);
		}

		// If this thread has the best RMS
		if (tIdx == bestThreadIdx)
		{
			printf("Thread %d: Line #: %d, LineIdx: %d, Global Best RMS: %f\n", bestThreadIdx, l, bestLocalIdx, bestRMSValue);
			for (int p = 0; p < lineSizes[bestLocalIdx]; p += 2)
			{
				unsigned short x = lineArray[lineStarts[bestLocalIdx] + p];
				unsigned short y = lineArray[lineStarts[bestLocalIdx] + p + 1];
				threadImg[x + y * imgSize] = 0;
				truthImg[x + y * imgSize] = min(truthImg[x + y * imgSize] + 40, 255);
			}
			finishedLines[bestLocalIdx] = true;

			finishedLinesAmt++;
			if (finishedLinesAmt == threadLineAmt)
			{
				delete[] lineStarts;
				delete[] lineSizes;
				delete[] finishedLines;
				return;
			}
		}

		cuda_SYNCTHREADS();
	}

	delete[] lineStarts;
	delete[] lineSizes;
	delete[] finishedLines;
}

void runCUDAThreader()
{
	size_t imgSize;
	const auto originalImageLarge = utils::prepareImage("res/huge_walter.png", imgSize);
	const std::vector<unsigned char> originalImage(originalImageLarge.begin(), originalImageLarge.end());

	const auto pins = utils::generatePins(imgSize, NUM_PINS);

	std::vector<unsigned int> linePtrOffsets(UNIQUE_LINE_NUMBER);
	auto lines = calculateLines(pins, imgSize, linePtrOffsets);

	unsigned char* threadImageGPU;
	unsigned char* truthImageGPU;
	unsigned short* linesArrayGPU;
	unsigned int* linePtrOffsetsGPU;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaErrorCheck(cudaSetDevice(0));

	// Allocate GPU buffers for three vectors.
	cudaErrorCheck(cudaMalloc((void**)&truthImageGPU, imgSize * imgSize * sizeof(unsigned char)));

	cudaErrorCheck(cudaMalloc((void**)&threadImageGPU, imgSize * imgSize * sizeof(unsigned char)));
	cudaErrorCheck(cudaMemset(threadImageGPU, 255, imgSize * imgSize * sizeof(unsigned char)));

	cudaErrorCheck(cudaMalloc((void**)&linesArrayGPU,lines.size() * sizeof(unsigned short)));

	cudaErrorCheck(cudaMalloc((void**)&linePtrOffsetsGPU, UNIQUE_LINE_NUMBER * sizeof(unsigned int)));

	// Copy image frame from host memory to GPU buffers.
	cudaErrorCheck(
		cudaMemcpy(truthImageGPU, originalImage.data(), imgSize * imgSize * sizeof(unsigned char),
			cudaMemcpyHostToDevice));

	cudaErrorCheck(
		cudaMemcpy(linesArrayGPU, lines.data(), lines.size() * sizeof(unsigned short), cudaMemcpyHostToDevice));

	cudaErrorCheck(
		cudaMemcpy(linePtrOffsetsGPU, linePtrOffsets.data(), UNIQUE_LINE_NUMBER * sizeof(unsigned int),
			cudaMemcpyHostToDevice));


	// Ensures there is a section of shared memory for each thread
	unsigned int threadCount = std::min(UNIQUE_LINE_NUMBER, 1024);
	//threadCount = 100;

	printf("CUDA kernel starting with %d threads...\n", threadCount);


	evaluateLineKernel<<<1, threadCount>> >(threadImageGPU, truthImageGPU, linesArrayGPU, linePtrOffsetsGPU,
	                                                       NUM_LINES,
	                                                       imgSize);

	// Check for any errors launching the kernel
	cudaErrorCheck(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaErrorCheck(cudaDeviceSynchronize());

	// Copy output vector from GPU buffer to host memory.
	auto threadImageOutput = std::vector<unsigned char>(imgSize * imgSize);
	cudaErrorCheck(
		cudaMemcpy(threadImageOutput.data(), threadImageGPU, imgSize * imgSize * sizeof(unsigned char),
			cudaMemcpyDeviceToHost));

	auto truthImageOutput = std::vector<unsigned char>(imgSize * imgSize);
	cudaErrorCheck(
		cudaMemcpy(truthImageOutput.data(), truthImageGPU, imgSize * imgSize * sizeof(unsigned char),
			cudaMemcpyDeviceToHost));

	utils::writePPM("output/output_cuda.ppm", utils::convert1c3c(threadImageOutput.data(), imgSize, imgSize).data(),
	                imgSize, imgSize);
	utils::writePPM("output/truth_cuda.ppm", utils::convert1c3c(truthImageOutput.data(), imgSize, imgSize).data(),
	                imgSize, imgSize);

	cudaFree(threadImageGPU);
	cudaFree(truthImageGPU);
	cudaFree(linesArrayGPU);
	cudaFree(linePtrOffsetsGPU);
}
