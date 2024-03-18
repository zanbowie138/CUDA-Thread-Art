#include <chrono>

#include "config.h"
#include "cuda_threader.cuh"
#include "cpu_threader.h"
#include "image_viewer.h"

#define PREVIEW_IMAGE 1

int main(int argc, char* argv[])
{
	std::cout << "Pin Number: " << NUM_PINS << std::endl;
	std::cout << "Unique Lines: " << UNIQUE_LINE_NUMBER << std::endl;
	std::cout << "Line Count: " << NUM_LINES << std::endl;

	std::cout << "Running with " << (RUN_WITH_CUDA ? "CUDA" : "CPU") << " threader...\n" << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	std::vector<unsigned char> outputImage;
	unsigned int imgSize;
	if (RUN_WITH_CUDA) { outputImage = runCUDAThreader(imgSize); }
	else { outputImage = runCPUThreader(imgSize); }

	std::cout << (RUN_WITH_CUDA ? "CUDA" : "CPU") << " threader finished" << std::endl;
	std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::high_resolution_clock::now() - start).count() << "ms" << std::endl;

	if (PREVIEW_IMAGE)
	{
		ImageViewer((std::string(RUN_WITH_CUDA ? "CUDA" : "CPU") + std::string(" Image")).c_str(), outputImage,
		            imgSize);
	}

	return 0;
}
