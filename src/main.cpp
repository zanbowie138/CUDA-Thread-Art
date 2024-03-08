#define RUN_WITH_CUDA 1

#include "cuda_threader.cuh"
#include "cpu_threader.h"


int main()
{
	std::cout << "Running with " << (RUN_WITH_CUDA ? "CUDA" : "CPU") << " threader" << std::endl;
	if (RUN_WITH_CUDA)
	{
		runCUDAThreader();
	}
	else
	{
		runCPUThreader();
	}
	return 0;
}
