#include <iostream>
#include <string>

#include "grayscaleCPU.h"
#include "grayscaleGPU.cuh"

#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 16

enum Type
{
    CPU,
    GPU
};

void CheckArguments(int argc, char **argv, Type &type)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image file | -h | --help>" << " <--cpu | --gpu>" << std::endl;
        exit(1);
    }
    else if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")
    {
        std::cout << "Usage: " << argv[0] << " <image file | -h | --help>" << " <--cpu | --gpu>" << std::endl;
        exit(0);
    }

    std::string arg = argv[2];
    if (arg == "--cpu")
    {
        type = CPU;
    }
    else if (arg == "--gpu")
    {
        type = GPU;
    }
    else
    {
        std::cerr << "Usage: " << argv[0] << " <image file | -h | --help>" << " <--cpu | --gpu>" << std::endl;
        exit(1);
    }
}

int main(int argc, char **argv)
{
    // ----------------SETUP---------------- //

    // Check for correct number of arguments
    Type type;
    CheckArguments(argc, argv, type);

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

    switch (type)
    {
    case CPU:
        std::cout << "Converting image to grayscale using CPU..." << std::endl;
        ConvertToGrayscaleCPU(image.data, image.rows, image.cols);
        break;
    case GPU:
        std::cout << "Converting image to grayscale using GPU..." << std::endl;
        ConvertToGrayscaleGPU(image.data, image.rows, image.cols, BLOCK_SIZE);
        break;
    default:
        break;
    }

    // ----------------OUTPUT---------------- //

    std::cout << "Writing image to file..." << std::endl;

    std::string path = argv[1];
    path = path.substr(0, path.find_last_of('/')) + "/results/";
    path += "grayscale_" + std::string(argv[1]).substr(std::string(argv[1]).find_last_of('/') + 1);
    std::cout << "Writing to " << path << std::endl;
    cv::imwrite(path, image);

    // write image result in a folder results
    

    // ----------------CLEANUP---------------- //

    image.release();

    std::cout << "Image processing complete!" << std::endl;
    return 0;
}