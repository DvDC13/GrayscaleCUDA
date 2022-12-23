#include <iostream>

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

    return 0;
}