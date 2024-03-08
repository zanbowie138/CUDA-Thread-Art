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

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

struct Point
{
    unsigned short x;
    unsigned short y;
};

static std::vector<Point> getLinePoints(Point start, Point end, int lineWidth, int size) {
    std::vector<Point> points;

    int dx = abs(end.x - start.x);
    int dy = abs(end.y - start.y);

    int sx = (start.x < end.x) ? 1 : -1;
    int sy = (start.y < end.y) ? 1 : -1;

    int err = dx - dy;

    while (true) {
        points.push_back({ start.x, start.y });

        // Add points for the line width
        for (short i = -lineWidth / 2; i <= lineWidth / 2; ++i) {
            if (abs(dx) > abs(dy)) {
                if (start.y + i >= 0 && start.y + i < size) {
                    points.push_back({ start.x, static_cast<unsigned short>(start.y + i) });
                }
            }
            else {
                if (start.x + i >= 0 && start.x + i < size) {
                    points.push_back({ static_cast<unsigned short>(start.x + i), start.y });
                }
            }
        }

        if (start.x == end.x && start.y == end.y) break;

        int e2 = 2 * err;

        if (e2 > -dy) {
            err -= dy;
            start.x += sx;
        }

        if (e2 < dx) {
            err += dx;
            start.y += sy;
        }
    }

    return points;
}

static std::vector<unsigned short> calculateLines(const std::vector<Point>& pins, int imgSize, unsigned int* &lineEndingIdx)
{
    lineEndingIdx = new unsigned int[UNIQUE_LINE_NUMBER];

    std::vector<std::vector<Point>> lines(UNIQUE_LINE_NUMBER);
    size_t pointAmount = 0;
    for (size_t i = 0; i < NUM_PINS - 1; i++) {
        for (size_t j = i + 1; j < NUM_PINS; j++) {
            const size_t lineIdx = (NUM_PINS * i - i * (i + 1) / 2) + j - (i + 1);
            lines[lineIdx] = getLinePoints(pins[i], pins[j], LINE_WIDTH, imgSize);
            pointAmount += lines[lineIdx].size();
        }
    }

    auto linesArray = std::vector<unsigned short>(pointAmount * 2);
    for (size_t i = 0; i < lines.size(); i++) {
        std::vector<Point> line = lines[i];
        if (i != 0) {
            lineEndingIdx[i] = lineEndingIdx[i - 1] + line.size();
            memcpy(&(linesArray[lineEndingIdx[i - 1] * 2 + 1]), line.data(), line.size() * sizeof(Point));
        }
        else
        {
	        lineEndingIdx[i] = line.size();
            memcpy(linesArray.data(), line.data(), line.size() * sizeof(Point));
        }
	}
    return linesArray;
}

static std::vector<Point> generatePins(size_t size, int pin_num) {
    std::vector<Point> pins(pin_num);
    float radius = size / 2.0f - 1;
    float center = size / 2.0f;
    float angleStep = 2.0f * 3.14159 / static_cast<float>(pin_num);

    for (int i = 0; i < pin_num; i++) {
        float angle = i * angleStep;
        pins[i].x = center + radius * cos(angle);
        pins[i].y = center + radius * sin(angle);
    }

    return pins;
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
            //printf("lineIdx: %d\n", lineIdx);
            lineStart = lineEndingIdx[lineIdx - 1] * 2 + 1;
            lineSize = (lineEndingIdx[lineIdx] - lineEndingIdx[lineIdx - 1]) * 2;
        }
        else
        {
            printf("lineIdx: %d\n", lineIdx);
            lineStart = 0;
            lineSize = lineEndingIdx[0] * 2;
            printf("lineSize: %d\n", lineSize);
        }

        for (int j = 0; j < lineSize; j += 2)
        {
            unsigned short x = lineArray[lineStart + j];
            unsigned short y = lineArray[lineStart + j + 1];
            threadImg[x + y * imgSize] = 255;
        }
    }
}


void runCUDAThreader()
{
    size_t imgSize;
    const auto originalImageLarge = utils::prepareImage("res/huge_walter.png", imgSize);
    const std::vector<unsigned char> originalImage(originalImageLarge.begin(), originalImageLarge.end());

	const auto pins = generatePins(imgSize, NUM_PINS);

	unsigned int* linePtrOffsets;
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

    cudaErrorCheck(cudaMemset(threadImageGPU, 0, imgSize * imgSize * sizeof(unsigned char)));

    cudaErrorCheck(cudaMalloc((void**)&linesArrayGPU,lines.size() * sizeof(unsigned short)));

    cudaErrorCheck(cudaMalloc((void**)&linePtrOffsetsGPU, UNIQUE_LINE_NUMBER * sizeof(unsigned int)));

    // Copy image frame from host memory to GPU buffers.
    cudaErrorCheck(cudaMemcpy(threadImageGPU, originalImage.data(), imgSize * imgSize * sizeof(unsigned char), cudaMemcpyHostToDevice));

	cudaErrorCheck(cudaMemcpy(linesArrayGPU, lines.data(), lines.size() * sizeof(unsigned short), cudaMemcpyHostToDevice));

	cudaErrorCheck(cudaMemcpy(linePtrOffsetsGPU, linePtrOffsets, UNIQUE_LINE_NUMBER * sizeof(unsigned int), cudaMemcpyHostToDevice));
    

	std::cout << "CUDA threader started" << std::endl;
	std::cout << "Unique Lines: " << UNIQUE_LINE_NUMBER << std::endl;
    //delete[] linePtrOffsets;

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

    std::cout << "CUDA threader finished" << std::endl;

    cudaFree(threadImageGPU);
    cudaFree(truthImageGPU);
    cudaFree(linesArrayGPU);
    cudaFree(linePtrOffsetsGPU);
}

void addVectors()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return;
    }
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
