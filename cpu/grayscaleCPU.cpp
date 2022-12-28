#include "grayscaleCPU.h"
#include "pixel.h"

#define channels 3

void ConvertToGrayscaleCPU(unsigned char* imageRGBA, int width, int height)
{
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
}