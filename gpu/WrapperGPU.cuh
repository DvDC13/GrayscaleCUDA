#pragma once

#include "Error.cuh"

void cudaXMalloc(void** ptr, size_t size) { CUDA_ERROR_CHECK(cudaMalloc(ptr, size)); }

void cudaXFree(void* ptr) { CUDA_ERROR_CHECK(cudaFree(ptr)); }

void cudaXMemcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) { CUDA_ERROR_CHECK(cudaMemcpy(dst, src, size, kind)); }

void cudaXDeviceSynchronize() { CUDA_ERROR_CHECK(cudaDeviceSynchronize()); }

void cudaXDeviceReset() { CUDA_ERROR_CHECK(cudaDeviceReset()); }