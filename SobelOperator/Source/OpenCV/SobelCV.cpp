#include "../../Headers/OpenCV/SobelCV.hpp"

cv::Mat input_image, blur_image, sobel_image;
cv::String cv_input_file, cv_output_file;
StopWatchInterface *cvTimer = nullptr;

void sobelCV(int image_number)
{
    switch(image_number)
    {
        case 1:
            cv_input_file = "Images\\Input\\img1.jpg";
            cv_output_file = "Images\\Output\\Sobel-CV-1.jpg";
        break;
        case 2:
            cv_input_file = "Images\\Input\\img2.jpg";
            cv_output_file = "Images\\Output\\Sobel-CV-2.jpg";
        break;
        case 3:
            cv_input_file = "Images\\Input\\img3.jpg";
            cv_output_file = "Images\\Output\\Sobel-CV-3.jpg";
        break;
        case 4:
            cv_input_file = "Images\\Input\\img4.jpg";
            cv_output_file = "Images\\Output\\Sobel-CV-4.jpg";
        break;
        case 5:
            cv_input_file = "Images\\Input\\img5.jpg";
            cv_output_file = "Images\\Output\\Sobel-CV-5.jpg";
        break;
        default:
            std::cerr << std::endl << "Error ... Please Enter Valid Number ... Exiting !!!" << std::endl;
            cvCleanup();
            exit(EXIT_FAILURE);
        break;
    }

    sdkCreateTimer(&cvTimer);
    
    input_image = cv::imread(cv_input_file, cv::IMREAD_GRAYSCALE);

    sdkStartTimer(&cvTimer);
    cv::GaussianBlur(input_image, blur_image, cv::Size(3, 3), 0);
    cv::Sobel(blur_image, sobel_image, CV_64F, 1, 1, 5);
    sdkStopTimer(&cvTimer);

    std::cout << std::endl << "Time for Sobel Operator using OpenCV (CPU) : " << sdkGetTimerValue(&cvTimer) << " ms" << std::endl;

    cv::imwrite(cv_output_file, sobel_image);

    cvCleanup();
}

void cvCleanup(void)
{
    if (cvTimer)
    {
        sdkDeleteTimer(&cvTimer);
        cvTimer = nullptr;
    }

    sobel_image.release();
    blur_image.release();
    input_image.release();
}
