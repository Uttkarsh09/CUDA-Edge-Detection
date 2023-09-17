#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_timer.h"

#define BLOCK_SIZE   32
#define GRID_SIZE    128
#define KERNEL_SIZE  3

__global__ void sobelFilter(unsigned char *input_image, unsigned char *output_image, unsigned int image_width, unsigned int image_height)
{
    int sobel_x[KERNEL_SIZE][KERNEL_SIZE] = {
        { -1, 0, 1 },
        { -2, 0, 2 },
        { -1, 0, 1 }
    };

    int sobel_y[KERNEL_SIZE][KERNEL_SIZE] = {
        { -1, -2, -1 },
        { 0, 0, 0 },
        { 1, 2, 1 }
    };

    int num_rows = blockIdx.x * blockDim.x + threadIdx.x;
    int num_columns = blockIdx.y * blockDim.y + threadIdx.y;

    int index = (num_rows * image_width) + num_columns;

    if ((num_columns < (image_width - 1)) && (num_rows < (image_height - 1)))
    {
        float gradient_x =  (input_image[index] * sobel_x[0][0]) + (input_image[index + 1] * sobel_x[0][1]) + (input_image[index + 2] * sobel_x[0][2]) +
                            (input_image[index] * sobel_x[1][0]) + (input_image[index + 1] * sobel_x[1][1]) + (input_image[index + 2] * sobel_x[1][2]) +
                            (input_image[index] * sobel_x[2][0]) + (input_image[index + 1] * sobel_x[2][1]) + (input_image[index + 2] * sobel_x[2][2]);

        float gradient_y =  (input_image[index] * sobel_y[0][0]) + (input_image[index + 1] * sobel_y[0][1]) + (input_image[index + 2] * sobel_y[0][2]) +
                            (input_image[index] * sobel_y[1][0]) + (input_image[index + 1] * sobel_y[1][1]) + (input_image[index + 2] * sobel_y[1][2]) +
                            (input_image[index] * sobel_y[2][0]) + (input_image[index + 1] * sobel_y[2][1]) + (input_image[index + 2] * sobel_y[2][2]);

        float gradient = sqrtf(gradient_x * gradient_x + gradient_y * gradient_y);

        if (gradient > 255)
            gradient = 255;

        if (gradient < 0)
            gradient = 0;

        __syncthreads();

        output_image[index] = gradient;
    }
}

void sobelGPU(cv::Mat *inputImage, cv::Mat *outputImage)
{
    unsigned char *deviceInput = nullptr, *deviceOutput = nullptr;
    cudaError_t result;
    StopWatchInterface *timer = nullptr;

    int imageSize = inputImage->rows * inputImage->cols * sizeof(unsigned char);

    sdkCreateTimer(&timer);

    result = cudaMalloc((void **)&deviceInput, imageSize);
    if (result != CUDA_SUCCESS)
    {
        std::cerr << std::endl << "cudaMalloc() Failed For Input Image ... Exiting !!!" << std::endl;
        exit(EXIT_FAILURE);
    }

    result = cudaMalloc((void **)&deviceOutput, imageSize);
    if (result != CUDA_SUCCESS)
    {
        std::cerr << std::endl << "cudaMalloc() Failed For Output Image ... Exiting !!!" << std::endl;
        exit(EXIT_FAILURE);
    }

    result = cudaMemcpy(deviceInput, inputImage->data, imageSize, cudaMemcpyHostToDevice);
    if (result != CUDA_SUCCESS)
    {
        std::cerr << std::endl << "cudaMemcpy() Failed For Input Image ... Exiting !!!" << std::endl;
        exit(EXIT_FAILURE);
    }

    const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 numBlocks(inputImage->cols, inputImage->rows);

    sdkStartTimer(&timer);
    sobelFilter<<<numBlocks, threadsPerBlock>>>(deviceInput, deviceOutput, inputImage->cols, inputImage->rows);
    sdkStopTimer(&timer);

    std::cout << std::endl << "Time for Sobel Operator using CUDA : " << sdkGetTimerValue(&timer) << " ms" << std::endl;

    result = cudaMemcpy(outputImage->data, deviceOutput, imageSize, cudaMemcpyDeviceToHost);
    if (result != CUDA_SUCCESS)
    {
        std::cerr << std::endl << "cudaMemcpy() Failed For Output Image ... Exiting !!!" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (deviceOutput)
    {
        cudaFree(deviceOutput);
        deviceOutput = nullptr;
    }

    if (deviceInput)
    {
        cudaFree(deviceInput);
        deviceInput = nullptr;
    }

    if (timer)
    {
        sdkDeleteTimer(&timer);
        timer = nullptr;
    }
}

int main(void)
{
    cv::Mat input_image, output_image;

    input_image = cv::imread("Images\\Input\\car-ai.jpg", cv::IMREAD_GRAYSCALE);
    output_image = input_image.clone();

    sobelGPU(&input_image, &output_image);

    cv::imwrite("Images\\Output\\Sobel-CUDA.jpg", output_image);

    input_image.release();
    output_image.release();
    
    return 0;
}
