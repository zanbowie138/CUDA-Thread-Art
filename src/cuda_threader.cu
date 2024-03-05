#pragma once

#include "cuda_threader.cuh"


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


void runCUDAThreader()
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

 //   int width, height, bpp;

 //   uint8_t* rgb_image = stbi_load("res/huge_walter.png", &width, &height, &bpp, 3);

 //   const auto gray_image = convertToGrayscale(rgb_image, width, height);

 //   stbi_image_free(rgb_image);

 //   auto cropped_image = cropImageToSquare(gray_image.data(), width, height);
 //   auto thread_image = std::vector<uint8_t>(cropped_image.size(), 255);
 //   auto compare_image = cropped_image;
 //   size_t size = std::min(width, height);
 //   auto pins = generatePins(size, NUM_PINS);
 //   std::vector<uint8_t> pin_image = cropped_image; // Copy the cropped image
 //   for (const auto& pin : pins) {
 //       pin_image[pin.y * size + pin.x] = 255; // Set the pixel at the pin's coordinates to white
 //   }
 //   bool linesDrawn[UNIQUE_LINE_NUMBER] = {false};
 //   std::vector<std::vector<Point>> lines(UNIQUE_LINE_NUMBER);
 //   std::cout << "UNIQUE_LINE_NUMBER: " << UNIQUE_LINE_NUMBER << "\n";
 //   assert(LINES <= UNIQUE_LINE_NUMBER && "Too many lines. Max is: " + UNIQUE_LINE_NUMBER);
 //   std::cout << "LINES: " << LINES << "\n";
 //   

 //   for (size_t l = 0; l < LINES; l++) {
 //       double bestError = std::numeric_limits<double>::max();
 //       size_t bestLine = 0;
 //       // 3 2 1
 //       // 0     1   2
 //       // 1 2 3 2 3 3
 //       for (size_t i = 0; i < NUM_PINS-1; i++) {
 //           for (size_t j = i + 1; j < NUM_PINS; j++) {
 //               const size_t lineIdx = (NUM_PINS * i - i * (i + 1) / 2) + j - (i + 1);
 //               if (l == 0) { lines[lineIdx] = getLinePoints(pins[i], pins[j], LINE_WIDTH, size); }
 //               //std::cout << "Line: " << lineIdx << "\n";
 //               if (linesDrawn[lineIdx]) { continue; }
 //               double error = calculateRMSError(compare_image, thread_image, size, lines[lineIdx]);
 //               if (error < bestError && !linesDrawn[lineIdx]) {
	//				bestError = error;
	//				bestLine = lineIdx;
	//			}
 //           }
	//	}
 //       assert(linesDrawn[bestLine] == false && "Line already drawn");
 //       linesDrawn[bestLine] = true;
 //       for (const auto& point : lines[bestLine]) {
	//		thread_image[point.y * size + point.x] = 0;
 //           compare_image[point.y * size + point.x] = std::min(compare_image[point.y * size + point.x] + 40, 255);

	//	}
 //       std::cout << "Best line: " << bestLine << " with error: " << bestError << "\n";
	//}

 //   writePPM("output/original.ppm", convert1c3c(pin_image.data(), size, size).data(), size, size);
 //   writePPM("output/output.ppm", convert1c3c(thread_image.data(), size, size).data(), size, size);
 //   writePPM("output/compare_img.ppm", convert1c3c(compare_image.data(), size, size).data(), size, size);

 //   std::cout << "RMS: " << calculateRMSError(cropped_image, thread_image, size/2-1, size) << "\n";
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
