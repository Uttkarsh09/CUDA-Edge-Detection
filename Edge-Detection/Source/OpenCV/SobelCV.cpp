#include "../../Headers/OpenCV/SobelCV.hpp"

cv::Mat input_img, blur_image, sobel_image;
cv::String sobel_cv_input_file, sobel_cv_output_file;
StopWatchInterface *sobelCVTimer = nullptr;

void sobelCV(int image_number)
{
    switch(image_number)
    {
        case 1:
            sobel_cv_input_file = "Images\\Input\\img1.jpg";
            sobel_cv_output_file = "Images\\Output\\Sobel-CV-1.jpg";
        break;
        case 2:
            sobel_cv_input_file = "Images\\Input\\img2.jpg";
            sobel_cv_output_file = "Images\\Output\\Sobel-CV-2.jpg";
        break;
        case 3:
            sobel_cv_input_file = "Images\\Input\\img3.jpg";
            sobel_cv_output_file = "Images\\Output\\Sobel-CV-3.jpg";
        break;
        case 4:
            sobel_cv_input_file = "Images\\Input\\img4.jpg";
            sobel_cv_output_file = "Images\\Output\\Sobel-CV-4.jpg";
        break;
        case 5:
            sobel_cv_input_file = "Images\\Input\\img5.jpg";
            sobel_cv_output_file = "Images\\Output\\Sobel-CV-5.jpg";
        break;
        default:
            std::cerr << std::endl << "Error ... Please Enter Valid Number ... Exiting !!!" << std::endl;
            cvCleanup();
            exit(EXIT_FAILURE);
        break;
    }

    sdkCreateTimer(&sobelCVTimer);
    
    input_img = cv::imread(sobel_cv_input_file, cv::IMREAD_GRAYSCALE);

    sdkStartTimer(&sobelCVTimer);
    cv::GaussianBlur(input_img, blur_image, cv::Size(3, 3), 0);
    cv::Sobel(blur_image, sobel_image, CV_64F, 1, 1, 5);
    sdkStopTimer(&sobelCVTimer);

    std::cout << std::endl << "Time for Sobel Operator using OpenCV (CPU) : " << sdkGetTimerValue(&sobelCVTimer) << " ms" << std::endl;

    cv::imwrite(sobel_cv_output_file, sobel_image);

    cvCleanup();
}

void cvCleanup(void)
{
    if (sobelCVTimer)
    {
        sdkDeleteTimer(&sobelCVTimer);
        sobelCVTimer = nullptr;
    }

    sobel_image.release();
    blur_image.release();
    input_img.release();
}
