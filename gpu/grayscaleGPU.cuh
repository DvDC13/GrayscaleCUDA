#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void ConvertToGrayscaleGPU(unsigned char* image, int rows, int cols, int blockSize);