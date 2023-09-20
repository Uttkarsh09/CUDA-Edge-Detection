#include <iostream>
#include <cstdlib>

#include "../Headers/OpenCV/SobelCV.hpp"
#include "../Headers/CUDA/SobelCUDA.cuh"

int main(int argc, char* argv[])
{
    sobelCV(atoi(argv[1]));

    sobelCUDA(atoi(argv[1]));

    exit(EXIT_SUCCESS);
}
