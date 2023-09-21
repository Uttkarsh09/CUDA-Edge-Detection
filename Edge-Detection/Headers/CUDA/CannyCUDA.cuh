#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef HELPER_TIMER_H
#define HELPER_TIMER_H
    #include "../helper_timer.h"
#endif

#ifndef NOMINMAX
    #define NOMINMAX
#endif

#define THREADS_PER_BLOCK     1024
#define KERNEL_SIZE           3
#define GRID                  0
#define CUDA_THRESHOLD        30

void cannyCUDA(int);
void cuda_canny_mem_alloc(void**, size_t);
void cuda_canny_mem_copy(void*, const void*, size_t, cudaMemcpyKind);
void cuda_canny_mem_free(void*);
void cannyCleanup(void);
void runCannyOperator(uint8_t*, uint8_t*);
