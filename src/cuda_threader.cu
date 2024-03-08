#include "cuda_threader.cuh"

#include "utils.h"

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void evaluateLineKernel(unsigned char* threadImg, const unsigned char* truthImg, const unsigned short* lineArray, const unsigned int* lineEndingIdx, unsigned int uniqueLineAmt, unsigned int imgSize)
{
    const unsigned int idx = threadIdx.x;
    if (idx >= UNIQUE_LINE_NUMBER) { return; }

    const unsigned int linesPerThread = static_cast<unsigned int>(ceil(static_cast<double>(uniqueLineAmt) / blockDim.x));
    unsigned int lineAmt;
    if (idx != blockDim.x || uniqueLineAmt % blockDim.x == 0)
    {
        lineAmt = linesPerThread;
    }
    else
    {
        lineAmt = uniqueLineAmt % blockDim.x;
    }

    for (int i = 0; i < lineAmt; i++)
    {
        unsigned int lineIdx = idx * linesPerThread + i;
        unsigned int lineStart;
        unsigned int lineSize;

        if (lineIdx != 0)
        {
            lineStart = lineEndingIdx[lineIdx - 1] * 2 + 1;
            lineSize = (lineEndingIdx[lineIdx] - lineEndingIdx[lineIdx - 1]) * 2;
        }
        else
        {
            lineStart = 0;
            lineSize = lineEndingIdx[0] * 2;
        }

        for (int j = 0; j < lineSize; j += 2)
        {
            unsigned short x = lineArray[lineStart + j];
            unsigned short y = lineArray[lineStart + j + 1];
            threadImg[x + y * imgSize] = 0;
        }
    }
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
    cudaErrorCheck(cudaMemcpy(truthImageGPU, originalImage.data(), imgSize * imgSize * sizeof(unsigned char), cudaMemcpyHostToDevice));

	cudaErrorCheck(cudaMemcpy(linesArrayGPU, lines.data(), lines.size() * sizeof(unsigned short), cudaMemcpyHostToDevice));

	cudaErrorCheck(cudaMemcpy(linePtrOffsetsGPU, linePtrOffsets.data(), UNIQUE_LINE_NUMBER * sizeof(unsigned int), cudaMemcpyHostToDevice));

	std::cout << "CUDA kernel starting..." << std::endl;

    evaluateLineKernel<<<1, 1 >>>(threadImageGPU, truthImageGPU, linesArrayGPU, linePtrOffsetsGPU, UNIQUE_LINE_NUMBER, imgSize);

    // Check for any errors launching the kernel
    cudaErrorCheck(cudaGetLastError());

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaErrorCheck(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    auto threadImageOutput = std::vector<unsigned char>(imgSize * imgSize);
    cudaErrorCheck(cudaMemcpy(threadImageOutput.data(), threadImageGPU, imgSize * imgSize * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    auto truthImageOutput = std::vector<unsigned char>(imgSize * imgSize);
    cudaErrorCheck(cudaMemcpy(truthImageOutput.data(), truthImageGPU, imgSize * imgSize * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    utils::writePPM("output/output_cuda.ppm", utils::convert1c3c(threadImageOutput.data(), imgSize, imgSize).data(), imgSize, imgSize);
    utils::writePPM("output/truth_cuda.ppm", utils::convert1c3c(truthImageOutput.data(), imgSize, imgSize).data(), imgSize, imgSize);

    cudaFree(threadImageGPU);
    cudaFree(truthImageGPU);
    cudaFree(linesArrayGPU);
    cudaFree(linePtrOffsetsGPU);
}
