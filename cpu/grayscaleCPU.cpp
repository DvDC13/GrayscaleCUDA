#include "grayscaleCPU.h"
#include "pixel.h"

#include <iostream>
#include <chrono>

#define channels 3

void ConvertToGrayscaleCPU(unsigned char* imageRGBA, int width, int height)
{
    auto start = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;
            Pixel* pixel = reinterpret_cast<Pixel*>(&imageRGBA[index * channels]);

            unsigned char pixelGray = static_cast<unsigned char>(0.2126f * pixel->red + 0.7152f * pixel->green + 0.0722f * pixel->blue);

            pixel->red = pixelGray;
            pixel->green = pixelGray;
            pixel->blue = pixelGray;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "CPU grayscale conversion took " << elapsed.count() * 1e3 << " ms" << std::endl;
}