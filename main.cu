#include <iostream>
#include <string>

#include "stb_image.h"
#include "stb_image_write.h"

#include "grayscaleCPU.h"
#include "grayscaleGPU.cuh"

#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 16

int main(int argc, char **argv)
{
    // ----------------SETUP---------------- //

    // Check for correct number of arguments
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image file>" << std::endl;
        return 1;
    }

    std::cout << "Beginning image processing..." << std::endl;

    // Load image
    std::cout << "Loading image..." << std::endl;
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

    // Check if image loaded successfully    
    if (image.empty())
    {
        std::cerr << "Failed to load image file " << argv[1] << std::endl;
        return 1;
    }

    std::cout << "Image loaded successfully!" << std::endl;

    std::cout << "Image width: " << image.cols << std::endl;
    std::cout << "Image height: " << image.rows << std::endl;
    std::cout << "Number of channels: " << 3 << std::endl;

    // ----------------PROCESSING---------------- //

    std::cout << "Processing image..." << std::endl;
    std::cout << "Converting image to grayscale..." << std::endl;

    unsigned char *imageDataDevice = nullptr;
    std::cout << "Allocating memory on GPU..." << std::endl;
    cudaMalloc(&imageDataDevice, image.cols * image.rows * 3);
    std::cout << "Copying image data to GPU..." << std::endl;
    cudaMemcpy(imageDataDevice, image.data, image.cols * image.rows * 3, cudaMemcpyHostToDevice);
    
    std::cout << "Launching kernel..." << std::endl;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    std::cout << "Block size: " << block.x << "x" << block.y << std::endl;

    dim3 grid((image.cols + block.x - 1) / block.x, (image.rows + block.y - 1) / block.y, 1);
    std::cout << "Grid size: " << grid.x << "x" << grid.y << std::endl;

    ConvertToGrayscaleGPU<<<grid, block>>>(imageDataDevice);
    cudaDeviceSynchronize();

    std::cout << "Copying image data back to CPU..." << std::endl;
    cudaMemcpy(image.data, imageDataDevice, image.cols * image.rows * 3, cudaMemcpyDeviceToHost);
    std::cout << "Image processed successfully!" << std::endl;

    // ----------------OUTPUT---------------- //

    std::cout << "Writing image to file..." << std::endl;

    std::string outputFilename = argv[1];
    outputFilename = outputFilename.substr(0, outputFilename.find_last_of('.')) + "_output.";
    std::string extension = argv[1];
    extension = extension.substr(extension.find_last_of('.') + 1);
    outputFilename += extension;
    std::cout << "Output filename: " << outputFilename << std::endl;
    cv::imwrite(outputFilename, image);

    cudaFree(imageDataDevice);
    image.release();

    cudaDeviceReset();

    std::cout << "Done!" << std::endl;
    return 0;
}