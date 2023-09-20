#include <iostream>
#include <cstdlib>

#include "../Headers/CUDA/SobelCUDA.cuh"
#include "../Headers/OpenCV/SobelCV.hpp"

int main()
{
    cudaHello();

    opencvHello();

    exit(EXIT_SUCCESS);
}
