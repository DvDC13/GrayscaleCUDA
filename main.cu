#include <iostream>
#include <string>

#include "stb_image.h"
#include "stb_image_write.h"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image file>" << std::endl;
        return 1;
    }

    int width, height, nrChannels;
    unsigned char *imageData = stbi_load(argv[1], &width, &height, &nrChannels, 4);
    
    if (!imageData)
    {
        std::cerr << "Failed to load image file " << argv[1] << std::endl;
        return 1;
    }

    std::cout << "Image width: " << width << std::endl;
    std::cout << "Image height: " << height << std::endl;
    std::cout << "Number of channels: " << nrChannels << std::endl;







    std::cout << "Writing image to file..." << std::endl;
    std::string outputFileName = argv[1];
    outputFileName = outputFileName.substr(0, outputFileName.find_last_of('.')) + "_grayscale.png";
    stbi_write_png(outputFileName.c_str(), width, height, nrChannels, imageData, width * nrChannels);

    stbi_image_free(imageData);
    std::cout << "Done!" << std::endl;
    return 0;
}