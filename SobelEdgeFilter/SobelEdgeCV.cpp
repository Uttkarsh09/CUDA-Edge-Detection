#include <opencv2/opencv.hpp>
#include <iostream>
#include "helper_timer.h"

cv::Mat image, blur_image, sobelXY_image;
StopWatchInterface *timer = NULL;

int main(void)
{
    sdkCreateTimer(&timer);
    
    image = cv::imread("Images\\Input\\win4.jpg", cv::IMREAD_GRAYSCALE);

    sdkStartTimer(&timer);
    cv::GaussianBlur(image, blur_image, cv::Size(3, 3), 0);
    cv::Sobel(blur_image, sobelXY_image, CV_64F, 1, 1, 5);
    sdkStopTimer(&timer);

    std::cout << std::endl << "Time for Sobel Operator using OpenCV (CPU) : " << sdkGetTimerValue(&timer) << " ms" << std::endl;

    cv::imwrite("Images\\Output\\Sobel-CV.jpg", sobelXY_image);
    
    if (timer)
    {
        sdkDeleteTimer(&timer);
        timer = NULL;
    }

    sobelXY_image.release();
    blur_image.release();
    image.release();

    return 0;
}

