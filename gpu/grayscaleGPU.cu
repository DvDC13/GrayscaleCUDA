#include "grayscaleGPU.cuh"
#include "pixel.h"
#include "WrapperGPU.cuh"

#include <iostream>

#define channels 3

__global__ void ConvertToGrayscaleGPU(unsigned char* imageRGBA)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int gridIdx = x + y * blockDim.x * gridDim.x;

    Pixel* pixel = reinterpret_cast<Pixel*>(&imageRGBA[gridIdx * channels]);
    unsigned char pixelGray = static_cast<unsigned char>(0.2126f * pixel->red + 0.7152f * pixel->green + 0.0722f * pixel->blue);
    pixel->red = pixelGray;
    pixel->green = pixelGray;
    pixel->blue = pixelGray;
}

void ConvertToGrayscaleGPU(unsigned char* image, int rows, int cols, int blockSize)
{
    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Processing image..." << std::endl;
    std::cout << "Converting image to grayscale..." << std::endl;

    unsigned totalPixels = cols * rows;
    unsigned bytesSize = totalPixels * 3 * sizeof(unsigned char);

    // ----------------PROCESSING---------------- //

    unsigned char *imageDataDevice = nullptr;
    cudaXMalloc((void **)&imageDataDevice, bytesSize);
    std::cout << "Allocated " << bytesSize << " bytes on device" << std::endl;
    cudaXMemcpy(imageDataDevice, image, bytesSize, cudaMemcpyHostToDevice);
    std::cout << "Copied " << bytesSize << " bytes to device" << std::endl;
    
    dim3 block(blockSize, blockSize, 1);
    std::cout << "Block size: " << block.x << "x" << block.y << std::endl;

    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y, 1);
    std::cout << "Grid size: " << grid.x << "x" << grid.y << std::endl;

    cudaEventRecord(start);
    ConvertToGrayscaleGPU<<<grid, block>>>(imageDataDevice);
    cudaEventRecord(stop);
    cudaXDeviceSynchronize();

    cudaXMemcpy(image, imageDataDevice, bytesSize, cudaMemcpyDeviceToHost);

    std::cout << "Image processed successfully!" << std::endl;

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

   // ----------------TIMING---------------- //

    std::cout << std::endl << std::endl << "Kernel Time execution taken: " << milliseconds << " ms" << std::endl;
    std::cout << std::endl << std::endl;

    cudaXFree(imageDataDevice);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaXDeviceReset();
}