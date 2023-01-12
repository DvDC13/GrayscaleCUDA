#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>

#define CUDA_ERROR_CHECK { gpuAssert(__FILE__, __LINE__); }

inline void gpuAssert(const char *file, int line, bool abort=true)
{
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
