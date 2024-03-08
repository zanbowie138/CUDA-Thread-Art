#define RUN_WITH_CUDA 1

#include <chrono>

#include "cuda_threader.cuh"
#include "cpu_threader.h"


int main()
{
	auto start = std::chrono::high_resolution_clock::now();

	std::cout << "Pin Number: " << NUM_PINS << std::endl;
	std::cout << "Unique Lines: " << UNIQUE_LINE_NUMBER << std::endl;
	std::cout << "Line Count: " << NUM_LINES << std::endl;

	std::cout << "Running with " << (RUN_WITH_CUDA ? "CUDA" : "CPU") << " threader...\n" << std::endl;
	if (RUN_WITH_CUDA)
	{
		runCUDAThreader();
	}
	else
	{
		runCPUThreader();
	}

	std::cout << (RUN_WITH_CUDA ? "CUDA" : "CPU") << " threader finished" << std::endl;
	std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << "ms" << std::endl;
	return 0;
}
