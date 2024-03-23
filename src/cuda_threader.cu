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



__device__ bool isLineIntersect(unsigned int p1l1, unsigned int p2l1, unsigned int p1l2, unsigned int p2l2, unsigned int pinNum)
{
	struct Line {
		unsigned int p1;
		unsigned int p2;
	};

	Line l1 = { min(p1l1, p2l1), max(p1l1, p2l2) };
	Line l2 = { min(p1l2, p2l2), max(p1l2, p2l2) };

	unsigned int o = l1.p1;
	l1 = { 0, l1.p2 - o };
	l2 = { (l2.p1 - o) % pinNum, (l2.p2 - o) % pinNum };

	// If both pins of the second line are on the same side of the first line, they are not intersecting
	return l2.p1 < l1.p2 != l2.p2 < l1.p2;
}

__global__ void evaluateLineKernel(unsigned char* threadImg, unsigned char* truthImg,
                                   const unsigned short* lineArray, const unsigned int* lineEndingIdx, const unsigned int* pinIdx,
                                   unsigned int globalLineAmt, unsigned int imgSize)
{
	extern __shared__ SortUnion sharedMem[];

	const unsigned int tIdx = threadIdx.x;

	if (tIdx >= UNIQUE_LINE_NUMBER) { return; }

	const unsigned int baseLinesPerThread = static_cast<unsigned int>(
		floor(static_cast<double>(UNIQUE_LINE_NUMBER) / blockDim.x));
	const unsigned int remainder = UNIQUE_LINE_NUMBER % blockDim.x;

	// Amount of lines for this thread
	unsigned int threadLineAmt = baseLinesPerThread + (tIdx < remainder);

	const auto lineStarts = new unsigned int[threadLineAmt];
	const auto lineSizes = new unsigned int[threadLineAmt];
	const auto linePins = new unsigned int[threadLineAmt * 2];
	const auto cachedLineCosts = new float[threadLineAmt];

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

		linePins[i * 2] = pinIdx[lineIdx * 2];
		linePins[i * 2 + 1] = pinIdx[lineIdx * 2 + 1];
		//printf("Thread %d: Line %d: Start: %d, Size: %d\n", tIdx, lineIdx, lineStarts[i], lineSizes[i]);
	}

	cuda_SYNCTHREADS();

	__shared__ unsigned int lastLinePins[2];
	auto finishedLines = new bool[threadLineAmt];
	unsigned int finishedLinesAmt = 0;
	for (unsigned int l = 0; l < globalLineAmt; l++)
	{
		unsigned int bestLocalIdx = 9999;
		if (finishedLinesAmt != threadLineAmt) {
			float bestRMSError = 99999;
			unsigned int bestRMSIdx = 9999999;
			for (int j = 0; j < threadLineAmt; j += 1)
			{
				if (finishedLines[j]) { continue; }

				float error = 0;
				// If not the first line and the previous line and this one are not intersecting, use the cached error
				if (l != 0 && !isLineIntersect(lastLinePins[0], lastLinePins[1], linePins[j * 2], linePins[j * 2 + 1], NUM_PINS))
				{
					error = cachedLineCosts[j];
				} else
				{
					error = calculateRMSError(truthImg, imgSize, lineArray, lineStarts[j], lineSizes[j]);
					cachedLineCosts[j] = error;
				}
				

				// If error is better than the current best, update thread best
				if (error < bestRMSError)
				{
					bestRMSError = error;
					bestRMSIdx = tIdx;
					bestLocalIdx = j;
					if (error == 0) { break; }
					//printf("Thread %d: Best Intermediate RMS: %f , Best Intermediate RMS Index: %d\n", tIdx, bestRMS, bestLocalIdx);
				}
			}
			sharedMem[tIdx].floats[0] = bestRMSError;
			sharedMem[tIdx].ints[1] = bestRMSIdx;
		} else
		{
			sharedMem[tIdx].floats[0] = 99999;
			sharedMem[tIdx].ints[1] = 99999;
		}

		cuda_SYNCTHREADS();

		// Parallel reduction to find the best RMS
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tIdx < s) {
				const bool isBest = sharedMem[tIdx + s].floats[0] < sharedMem[tIdx].floats[0];
				sharedMem[tIdx] = isBest ? sharedMem[tIdx + s] : sharedMem[tIdx];
			}
			cuda_SYNCTHREADS();
		}

		cuda_SYNCTHREADS();

		if (finishedLinesAmt != threadLineAmt) {
			const SortUnion bestRMS = sharedMem[0];
			const float bestGlobalRMSValue = bestRMS.floats[0];
			const unsigned int bestGlobalThreadIdx = bestRMS.ints[1];

			/*if (bestRMSValue > bestRMS)
			{
				printf("ERROR: SORT FAILED... Best RMS: %f, Global Best RMS: %f\n", bestRMS, bestRMSValue);
			}*/

			// If this thread has the best RMS
			if (tIdx == bestGlobalThreadIdx)
			{
				printf("Thread %d: Line #: %d, LineIdx: %d, Global Best RMS: %f\n", tIdx, l, bestLocalIdx, bestGlobalRMSValue);
				for (int p = 0; p < lineSizes[bestLocalIdx]; p += 2)
				{
					unsigned short x = lineArray[lineStarts[bestLocalIdx] + p];
					unsigned short y = lineArray[lineStarts[bestLocalIdx] + p + 1];
					threadImg[x + y * imgSize] = max(threadImg[x + y * imgSize] - 100, 0);
					truthImg[x + y * imgSize] = min(truthImg[x + y * imgSize] + 20, 255);
				}

				lastLinePins[0] = linePins[bestLocalIdx * 2];
				lastLinePins[1] = linePins[bestLocalIdx * 2 + 1];

				finishedLines[bestLocalIdx] = true;
				finishedLinesAmt++;
			}
		}

		cuda_SYNCTHREADS();
	}

	delete[] lineStarts;
	delete[] lineSizes;
	delete[] finishedLines;
	delete[] cachedLineCosts;
}

std::vector<unsigned char> runCUDAThreader(unsigned int& size)
{
	size_t imgSize;
	const auto originalImage = utils::prepareImage("res/color_cat.png", imgSize);
	size = imgSize;

	const auto pins = utils::generatePins(imgSize, NUM_PINS);

	std::vector<unsigned int> linePtrOffsets(UNIQUE_LINE_NUMBER);
	std::vector<unsigned int> pinIdx(UNIQUE_LINE_NUMBER * 2);
	std::cout << "Calculating lines..." << std::endl;
	auto lines = calculateLines(pins, imgSize, linePtrOffsets, pinIdx);

	unsigned char* threadImageGPU;
	unsigned char* truthImageGPU;
	unsigned short* linesArrayGPU;
	unsigned int* linePtrOffsetsGPU;
	unsigned int* pinIdxGPU;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaErrorCheck(cudaSetDevice(0));

	// Allocate GPU buffers for three vectors.
	cudaErrorCheck(cudaMalloc((void**)&truthImageGPU, imgSize * imgSize * sizeof(unsigned char)));

	cudaErrorCheck(cudaMalloc((void**)&threadImageGPU, imgSize * imgSize * sizeof(unsigned char)));
	cudaErrorCheck(cudaMemset(threadImageGPU, 255, imgSize * imgSize * sizeof(unsigned char)));

	cudaErrorCheck(cudaMalloc((void**)&linesArrayGPU,lines.size() * sizeof(unsigned short)));

	cudaErrorCheck(cudaMalloc((void**)&linePtrOffsetsGPU, UNIQUE_LINE_NUMBER * sizeof(unsigned int)));

	cudaErrorCheck(cudaMalloc((void**)&pinIdxGPU, UNIQUE_LINE_NUMBER * 2 * sizeof(unsigned int)));

	// Copy image frame from host memory to GPU buffers.
	cudaErrorCheck(
		cudaMemcpy(truthImageGPU, originalImage.data(), imgSize * imgSize * sizeof(unsigned char),
			cudaMemcpyHostToDevice));

	cudaErrorCheck(
		cudaMemcpy(linesArrayGPU, lines.data(), lines.size() * sizeof(unsigned short), cudaMemcpyHostToDevice));

	cudaErrorCheck(
		cudaMemcpy(linePtrOffsetsGPU, linePtrOffsets.data(), UNIQUE_LINE_NUMBER * sizeof(unsigned int),
			cudaMemcpyHostToDevice));

	cudaErrorCheck(
		cudaMemcpy(pinIdxGPU, pinIdx.data(), UNIQUE_LINE_NUMBER * 2 * sizeof(unsigned int),
			cudaMemcpyHostToDevice));


	// Ensures there is a section of shared memory for each thread
	unsigned int threadCount = std::min(UNIQUE_LINE_NUMBER, 1024);
	//threadCount = 100;

	printf("CUDA kernel starting with %d threads...\n", threadCount);


	evaluateLineKernel<<<1, threadCount, threadCount * sizeof(SortUnion) >> >(threadImageGPU, truthImageGPU, linesArrayGPU, linePtrOffsetsGPU, pinIdxGPU,
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

	return threadImageOutput;
}
