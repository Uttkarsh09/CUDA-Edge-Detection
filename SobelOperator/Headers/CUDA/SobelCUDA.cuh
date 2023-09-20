#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../helper_timer.h"

#define BLOCK_SIZE            32
#define GRID_SIZE             128
#define SOBEL_KERNEL_SIZE     5
#define GAUSSIAN_KERNEL_SIZE  3

void sobelCUDA(int);
void cleanup();
void runSobelOperator(cv::Mat*, cv::Mat*);
