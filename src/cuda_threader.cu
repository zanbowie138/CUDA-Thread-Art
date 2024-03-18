#include "cuda_threader.cuh"

#include "utils.h"

__device__ float calculateRMSError(const unsigned char* truthImg,
                                   const unsigned int imgSize,
                                   const unsigned short* lineArray, const unsigned int lineStart,
                                   const unsigned int lineSize)
{
	float avg = 0.0;
	const float count = lineSize / 2;

	if (lineSize == 0) { printf("Empty RMS\n"); }

	for (int i = 0; i < lineSize; i += 2)
	{
		unsigned short x = lineArray[lineStart + i];
		unsigned short y = lineArray[lineStart + i + 1];
		float diff = truthImg[y * imgSize + x];
		avg += diff * diff / count;
	}

	return sqrtf(avg);
}

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

	if (tIdx >= UNIQUE_LINE_NUMBER) { return; }

	const unsigned int baseLinesPerThread = static_cast<unsigned int>(
		floor(static_cast<double>(UNIQUE_LINE_NUMBER) / blockDim.x));
	const unsigned int remainder = UNIQUE_LINE_NUMBER % blockDim.x;

	// Amount of lines for this thread
	unsigned int threadLineAmt = baseLinesPerThread + (tIdx < remainder);

	const auto lineStarts = new unsigned int[threadLineAmt];
	const auto lineSizes = new unsigned int[threadLineAmt];

	// Calculate the start and size of each line
	for (int i = 0; i < threadLineAmt; i++)
	{
		unsigned int lineIdx;
		if (tIdx < remainder) { lineIdx = tIdx * (baseLinesPerThread + 1) + i; }
		else { lineIdx = tIdx * baseLinesPerThread + i + remainder; }

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

	__shared__ SortUnion globalBestRMS;
	auto finishedLines = new bool[threadLineAmt];
	unsigned int finishedLinesAmt = 0;

	globalBestRMS.floats[0] = __float_as_uint(99999);
	globalBestRMS.ints[1] = 99999;

	for (unsigned int l = 0; l < globalLineAmt; l++)
	{
		//if (globalBestRMS.ints[1] != 99999) { printf("Error: Global Best did not reset correctly. \n"); }

		float bestRMS = 99999;
		unsigned int bestRMSIdx = 9999999;
		unsigned int bestLocalIdx = 9999;
		for (int j = 0; j < threadLineAmt; j += 1)
		{
			if (finishedLines[j]) { continue; }

			const float error = calculateRMSError(truthImg, imgSize, lineArray, lineStarts[j], lineSizes[j]);

			// If error is better than the current best, update thread best
			if (error < bestRMS)
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
		atomicMinIndex(&globalBestRMS.ulong, bestRMS, bestRMSIdx);

		const unsigned int bestThreadIdx = globalBestRMS.ints[1];
		const float bestRMSValue = globalBestRMS.floats[0];

		if (bestRMSValue > bestRMS)
		{
			printf("ERROR: SORT FAILED... Best RMS: %f, Global Best RMS: %f\n", bestRMS, bestRMSValue);
		}

		// If this thread has the best RMS
		if (tIdx == bestThreadIdx)
		{
			printf("Thread %d: Line #: %d, LineIdx: %d, Global Best RMS: %f\n", tIdx, l, bestLocalIdx, bestRMSValue);
			for (int p = 0; p < lineSizes[bestLocalIdx]; p += 2)
			{
				unsigned short x = lineArray[lineStarts[bestLocalIdx] + p];
				unsigned short y = lineArray[lineStarts[bestLocalIdx] + p + 1];
				threadImg[x + y * imgSize] = 0;
				truthImg[x + y * imgSize] = min(truthImg[x + y * imgSize] + 40, 255);
			}

			// Reset global best
			globalBestRMS.floats[0] = __float_as_uint(99999);
			globalBestRMS.ints[1] = 99999;

			finishedLines[bestLocalIdx] = true;
			if (++finishedLinesAmt == threadLineAmt) { break; }
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
	std::cout << "Calculating lines..." << std::endl;
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
