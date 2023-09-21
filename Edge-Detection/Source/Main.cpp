#include <iostream>
#include <cstdlib>

#include "../Headers/Header.hpp"

int main(int argc, char* argv[])
{
    sobelCV(atoi(argv[1]));

    sobelCUDA(atoi(argv[1]));

    cannyCV(atoi(argv[1]));

    cannyCUDA(atoi(argv[1]));

    exit(EXIT_SUCCESS);
}
