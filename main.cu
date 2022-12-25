#include <iostream>
#include <string>

#include "stb_image.h"
#include "stb_image_write.h"

#include "grayscaleCPU.h"
#include "grayscaleGPU.cuh"

#define BLOCK_SIZE 32

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image file>" << std::endl;
        return 1;
    }

    std::cout << "Beginning image processing..." << std::endl;

    int width, height, nrChannels;
    unsigned char *imageData = stbi_load(argv[1], &width, &height, &nrChannels, 4);
    
    if (!imageData)
    {
        std::cerr << "Failed to load image file " << argv[1] << std::endl;
        return 1;
    }

    std::cout << "Image loaded successfully!" << std::endl;

    std::cout << "Image width: " << width << std::endl;
    std::cout << "Image height: " << height << std::endl;
    std::cout << "Number of channels: " << nrChannels << std::endl;

    std::cout << "Processing image..." << std::endl;
    std::cout << "Converting image to grayscale..." << std::endl;

    unsigned char *imageDataDevice = nullptr;
    std::cout << "Allocating memory on GPU..." << std::endl;
    cudaMalloc(&imageDataDevice, width * height * nrChannels);
    std::cout << "Copying image data to GPU..." << std::endl;
    cudaMemcpy(imageDataDevice, imageData, width * height * nrChannels, cudaMemcpyHostToDevice);
    
    std::cout << "Launching kernel..." << std::endl;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    std::cout << "Block size: " << block.x << "x" << block.y << std::endl;
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    std::cout << "Grid size: " << grid.x << "x" << grid.y << std::endl;
    ConvertToGrayscaleGPU<<<grid, block>>>(imageDataDevice, nrChannels);
    cudaDeviceSynchronize();

    cudaMemcpy(imageData, imageDataDevice, width * height * nrChannels, cudaMemcpyDeviceToHost);
    std::cout << "Image processed successfully!" << std::endl;

    std::cout << "Writing image to file..." << std::endl;
    std::string outputFileName = argv[1];
    outputFileName = outputFileName.substr(0, outputFileName.find_last_of('.')) + "_grayscale.png";
    stbi_write_png(outputFileName.c_str(), width, height, nrChannels, imageData, width * nrChannels);

    cudaFree(imageDataDevice);
    stbi_image_free(imageData);

    cudaDeviceReset();

    std::cout << "Done!" << std::endl;
    return 0;
}