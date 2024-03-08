#pragma once

#include <cassert>
#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "config.h"

/*
 * CUDA Core Number: 3072
 * Streaming Multiprocessor (SM) Number: 24
 * Maximum number of threads per block: 1024
 * Maximum number of resident grids per device (Concurrent Kernel Execution): 128
 */

__global__ void evaluateLineKernel();

void runCUDAThreader();

