#include "grayscaleGPU.cuh"
#include "pixel.h"

__global__ void ConvertToGrayscaleGPU(unsigned char* imageRGBA, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int gridIdx = x * channels + y * blockDim.x * gridDim.x * channels;

    Pixel* pixel = reinterpret_cast<Pixel*>(&imageRGBA[gridIdx]);
    unsigned char pixelGray = static_cast<unsigned char>(0.2126f * pixel->red + 0.7152f * pixel->green + 0.0722f * pixel->blue);
    pixel->red = pixelGray;
    pixel->green = pixelGray;
    pixel->blue = pixelGray;
    pixel->alpha = 255;
}