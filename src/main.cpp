#define RUN_WITH_CUDA 0

#include "cpu_threader.h"

int main()
{
	if (RUN_WITH_CUDA)
	{
		// Run with CUDA
	}
	else
	{
		runCPUThreader();
	}
	return 0;
}
