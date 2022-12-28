#include <iostream>
#include <string>

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

    ConvertToGrayscaleGPU(image.data, image.rows, image.cols, BLOCK_SIZE);

    // ----------------OUTPUT---------------- //

    std::cout << "Writing image to file..." << std::endl;

    std::string outputFilename = argv[1];
    outputFilename = outputFilename.substr(0, outputFilename.find_last_of('.')) + "_output.";
    std::string extension = argv[1];
    extension = extension.substr(extension.find_last_of('.') + 1);
    outputFilename += extension;
    std::cout << "Output filename: " << outputFilename << std::endl;
    cv::imwrite(outputFilename, image);

     // ----------------CLEANUP---------------- //

    image.release();

    std::cout << "Image processing complete!" << std::endl;
    return 0;
}