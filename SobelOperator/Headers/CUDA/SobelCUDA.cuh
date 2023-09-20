#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef HELPER_TIMER_H
#define HELPER_TIMER_H
    #include "../helper_timer.h"
#endif

#ifndef NOMINMAX
    #define NOMINMAX
#endif


#define BLOCK_SIZE            32
#define GRID_SIZE             128
#define SOBEL_KERNEL_SIZE     5
#define GAUSSIAN_KERNEL_SIZE  3

void sobelCUDA(int);
void cudaCleanup();
void runSobelOperator(cv::Mat*, cv::Mat*);
