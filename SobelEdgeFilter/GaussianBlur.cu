#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_timer.h"

#define KERNEL_SIZE    5

__global__ void gaussianBlurKernel(unsigned char *input_image, unsigned char *output_image, int width, int height, float *kernel, int kernelSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float blurPixel = 0.0f;
        int kernelRadius = kernelSize / 2;

        for (int i = -kernelRadius; i <= kernelRadius; i++)
        {
            for (int j = -kernelRadius; j <= kernelRadius; j++)
            {
                int xOffset = x + i;
                int yOffset = y + j;

                if (xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height)
                {
                    int inputIndex = yOffset * width + xOffset;
                    int kernelIndex = (i + kernelRadius) * kernelSize + (j + kernelRadius);
                    blurPixel = blurPixel + static_cast<float>(input_image[inputIndex]) * kernel[kernelIndex];
                }
            }
        }

        output_image[y * width + x] = static_cast<unsigned char>(blurPixel);
    }
}

void gaussianBlurGPU(cv::Mat *inputImage, cv::Mat *outputImage)
{
    // Variable Declarations
    cudaError_t result;
    StopWatchInterface *timer = nullptr;
    unsigned char *deviceInput = nullptr, *deviceOutput = nullptr;
    float *hostKernel = nullptr, *deviceKernel = nullptr;
    float kernelSum = 0.0f;
    float sigma = 1.0f;

    int imageWidth = inputImage->cols;
    int imageHeight = inputImage->rows;
    int imageSize = imageHeight * imageWidth * sizeof(unsigned char);
    
    // Create Gaussian Kernel
    hostKernel = new float[KERNEL_SIZE * KERNEL_SIZE];
    int kernelRadius = KERNEL_SIZE / 2;

    for (int i = -kernelRadius; i <= kernelRadius; i++) 
    {
        for (int j = -kernelRadius; j <= kernelRadius; j++)
        {
            int index = (i + kernelRadius) * kernelRadius + (j + kernelRadius);
            hostKernel[index] = exp(-(i * i + j + j) / (2.0f * sigma * sigma));
            kernelSum = kernelSum + hostKernel[index];
        }
    }

    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++)
    {
        hostKernel[i] = hostKernel[i] / kernelSum;
    }

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

    result = cudaMalloc((void **)&deviceKernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    if (result != CUDA_SUCCESS)
    {
        std::cerr << std::endl << "cudaMalloc() Failed For Device Kernel ... Exiting !!!" << std::endl;
        exit(EXIT_FAILURE);
    }

    result = cudaMemcpy(deviceInput, inputImage->data, imageSize, cudaMemcpyHostToDevice);
    if (result != CUDA_SUCCESS)
    {
        std::cerr << std::endl << "cudaMemcpy() Failed For Input Image ... Exiting !!!" << std::endl;
        exit(EXIT_FAILURE);
    }

    result = cudaMemcpy(deviceKernel, hostKernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (result != CUDA_SUCCESS)
    {
        std::cerr << std::endl << "cudaMemcpy() Failed For Device Kernel ... Exiting !!!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Kernel Configuration
    dim3 dimBlock(16, 16);
    dim3 dimGrid((imageWidth + dimBlock.x - 1) / dimBlock.x, (imageHeight + dimBlock.y - 1) / dimBlock.y);

    sdkStartTimer(&timer);
    gaussianBlurKernel<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, imageWidth, imageHeight, deviceKernel, KERNEL_SIZE);
    sdkStopTimer(&timer);

    std::cout << std::endl << "Time for Gaussian using CUDA : " << sdkGetTimerValue(&timer) << " ms" << std::endl;

    result = cudaMemcpy(outputImage->data, deviceOutput, imageSize, cudaMemcpyDeviceToHost);
    if (result != CUDA_SUCCESS)
    {
        std::cerr << std::endl << "cudaMemcpy() Failed For Output Image ... Exiting !!!" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (deviceKernel)
    {
        cudaFree(deviceKernel);
        deviceKernel = nullptr;
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

    if (hostKernel)
    {
        delete[] hostKernel;
        hostKernel = nullptr;
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

    input_image = cv::imread("Images\\Input\\pikachu.jpg", cv::IMREAD_GRAYSCALE);
    output_image = input_image.clone();

    gaussianBlurGPU(&input_image, &output_image);

    output_image.convertTo(output_image, CV_8UC1);

    cv::imwrite("Images\\Output\\Gaussian-CUDA.jpg", output_image);

    input_image.release();
    output_image.release();
    
    exit(EXIT_SUCCESS);
}
