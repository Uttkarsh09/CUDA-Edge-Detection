#include <iostream>
#include <cstdlib>

#include "../Headers/OpenCV/CannyCV.hpp"
#include "../Headers/CUDA/CannyCUDA.cuh"

int main(int argc, char* argv[])
{
    cannyCV(atoi(argv[1]));

    cannyCUDA(atoi(argv[1]));

    exit(EXIT_SUCCESS);
}
