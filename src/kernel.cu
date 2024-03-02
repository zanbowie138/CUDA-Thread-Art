#define NUM_PINS 40
#define LINES 400
#define M_PI 3.1415926535
#define UNIQUE_LINE_NUMBER NUM_PINS * (NUM_PINS - 1) / 2

#include <cassert>
#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stb/stb_image.h>

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

struct Point
{
	int x, y;
};

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void writePPM(const char* filename, const uint8_t* data, int width, int height) {
	FILE* file = fopen(filename, "wb");
    if (file == NULL) { assert(false && "Error opening file"); }
	fprintf(file, "P6\n%d %d\n255\n", width, height);
	fwrite(data, sizeof(uint8_t), width * height * 3, file);
	fclose(file);
}

std::vector<uint8_t> convert1c3c(const uint8_t* image, int width, int height) {
	std::vector<uint8_t> three_channel_image(width * height * 3);
	for (size_t i = 0; i < width * height; i++) {
		three_channel_image[i * 3] = image[i];
		three_channel_image[i * 3 + 1] = image[i];
		three_channel_image[i * 3 + 2] = image[i];
	} 
	return three_channel_image;
}

std::vector<uint8_t> convertToGrayscale(const uint8_t* rgb_image, int width, int height) {
    std::vector<uint8_t> gray_image(width * height);
	for (size_t i = 0; i < width * height; i++) {
		int r = rgb_image[i * 3];
		int g = rgb_image[i * 3 + 1];
		int b = rgb_image[i * 3 + 2];
		int gray = 0.299 * r + 0.587 * g + 0.114 * b;
		gray_image[i] = gray;
	}
    return gray_image;
}

std::vector<uint8_t> cropImageToSquare(const uint8_t* image, int width, int height) {
    int size = std::min(width, height);
    std::vector<uint8_t> cropped_image(size * size);
    int startX = (width - size) / 2;
    int startY = (height - size) / 2;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            cropped_image[y * size + x] = image[(startY + y) * width + (startX + x)];
        }
    }
    return cropped_image;
}

std::vector<Point> generatePins(int size, int pin_num) {
    std::vector<Point> pins(pin_num);
    float radius = size / 2.0f - 1;
    float center = size / 2.0f;
    float angleStep = 2.0f * M_PI / static_cast<float>(pin_num);

    for (int i = 0; i < pin_num; i++) {
        float angle = i * angleStep;
        pins[i].x = center + radius * cos(angle);
        pins[i].y = center + radius * sin(angle);
    }

    return pins;
}

std::vector<Point> getLinePoints(Point start, Point end) {
    std::vector<Point> points;

    int dx = abs(end.x - start.x);
    int dy = abs(end.y - start.y);

    int sx = (start.x < end.x) ? 1 : -1;
    int sy = (start.y < end.y) ? 1 : -1;

    int err = dx - dy;

    while (true) {
        points.push_back(start);

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

double calculateRMSError(const std::vector<uint8_t>& img1, const std::vector<uint8_t>& img2, int radius, int size) {
    int center = size / 2;

    double sum = 0.0;
    int count = 0;

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int dx = x - center;
            int dy = y - center;
            if (dx * dx + dy * dy <= radius * radius) {
                int diff = img1[y * size + x] - img2[y * size + x];
                sum += diff * diff;
                ++count;
            }
        }
    }

    if (count == 0) {
        assert(false && "Empty RMS");
    }

    return std::sqrt(sum / count);
}

double calculateRMSError(const std::vector<uint8_t>& img1, const std::vector<uint8_t>& img2, size_t size, const std::vector<Point> coordinates) {
    double sum = 0.0;
    int count = 0;

    for (int y = 0; y < coordinates.size(); ++y) {
        for (int x = 0; x < coordinates.size(); ++x) {
        	int diff = img1[y * size + x] - img2[y * size + x];
            sum += diff * diff;
            ++count;
        }
    }

    if (count == 0) {
        assert(false && "Empty RMS");
    }

    return std::sqrt(sum / count);
}


int main()
{
    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    int width, height, bpp;

    uint8_t* rgb_image = stbi_load("res/cat.png", &width, &height, &bpp, 3);

    const auto gray_image = convertToGrayscale(rgb_image, width, height);

    stbi_image_free(rgb_image);

    auto cropped_image = cropImageToSquare(gray_image.data(), width, height);
    auto thread_image = std::vector<uint8_t>(cropped_image.size(), 255);
    size_t size = std::min(width, height);
    auto pins = generatePins(size, NUM_PINS);
    std::vector<uint8_t> image = cropped_image; // Copy the cropped image
    for (const auto& pin : pins) {
        image[pin.y * size + pin.x] = 255; // Set the pixel at the pin's coordinates to white
    }
    bool linesDrawn[UNIQUE_LINE_NUMBER] = {false};
    std::vector<std::vector<Point>> lines(UNIQUE_LINE_NUMBER);
    std::cout << "UNIQUE_LINE_NUMBER: " << UNIQUE_LINE_NUMBER << "\n";
    assert(LINES <= UNIQUE_LINE_NUMBER && "Too many lines. Max is: " + UNIQUE_LINE_NUMBER);
    std::cout << "LINES: " << LINES << "\n";
    

    for (size_t l = 0; l < LINES; l++) {
        double bestError = std::numeric_limits<double>::max();
        size_t bestLine = 0;
        size_t lineamt = 0;
        for (size_t i = 0; i < NUM_PINS-1; i++) {
            for (size_t j = i + 1; j < NUM_PINS; j++) {
                const size_t lineIdx = i * (i - 1) / 2 + j - 1;
                if (linesDrawn[lineIdx]) { continue;  }
                lineamt++;
                if (l == 0) { lines[lineIdx] = getLinePoints(pins[i], pins[j]); }
                double error = calculateRMSError(cropped_image, thread_image, size, lines[lineIdx]);
                if (error < bestError && !linesDrawn[lineIdx]) {
					bestError = error;
					bestLine = lineIdx;
				}
            }
            //std::cout << "pin #: " << i << "\n";
		}
        linesDrawn[bestLine] = true;
        for (const auto& point : lines[bestLine]) {
			thread_image[point.y * size + point.x] = 0;
		}
        std::cout << "Best line: " << bestLine << " with error: " << bestError << "\n";
	}

    writePPM("original.ppm", convert1c3c(image.data(), size, size).data(), size, size);
    writePPM("output.ppm", convert1c3c(thread_image.data(), size, size).data(), size, size);

    std::cout << "RMS: " << calculateRMSError(cropped_image, thread_image, size/2-1, size) << "\n";
    std::cout << "lineamount: " << UNIQUE_LINE_NUMBER << "\n";
	return 0;
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
