#include "../../Headers/OpenCV/SobelCV.hpp"

cv::Mat input_image, blur_image, sobel_image;
cv::String input_file, output_file;
StopWatchInterface *timer = nullptr;

void sobelCV(int image_number)
{
    switch(image_number)
    {
        case 1:
            input_file = "Images\\Input\\img1.jpg";
            output_file = "Images\\Output\\Sobel-CV-1.jpg";
        break;
        case 2:
            input_file = "Images\\Input\\img2.jpg";
            output_file = "Images\\Output\\Sobel-CV-2.jpg";
        break;
        case 3:
            input_file = "Images\\Input\\img3.jpg";
            output_file = "Images\\Output\\Sobel-CV-3.jpg";
        break;
        case 4:
            input_file = "Images\\Input\\img4.jpg";
            output_file = "Images\\Output\\Sobel-CV-4.jpg";
        break;
        case 5:
            input_file = "Images\\Input\\img5.jpg";
            output_file = "Images\\Output\\Sobel-CV-5.jpg";
        break;
        default:
            std::cerr << std::endl << "Error ... Please Enter Valid Number ... Exiting !!!" << std::endl;
            cleanup();
            exit(EXIT_FAILURE);
        break;
    }

    sdkCreateTimer(&timer);
    
    input_image = cv::imread(input_file, cv::IMREAD_GRAYSCALE);

    sdkStartTimer(&timer);
    cv::GaussianBlur(input_image, blur_image, cv::Size(3, 3), 0);
    cv::Sobel(blur_image, sobel_image, CV_64F, 1, 1, 5);
    sdkStopTimer(&timer);

    std::cout << std::endl << "Time for Sobel Operator using OpenCV (CPU) : " << sdkGetTimerValue(&timer) << " ms" << std::endl;

    cv::imwrite(output_file, sobel_image);

    cleanup();
}

void cleanup(void)
{
    if (timer)
    {
        sdkDeleteTimer(&timer);
        timer = nullptr;
    }

    sobel_image.release();
    blur_image.release();
    input_image.release();
}
