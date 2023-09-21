#include "../../Headers/OpenCV/CannyCV.hpp"

cv::Mat input_image, output_image;
cv::String canny_cv_input_file, canny_cv_output_file;
StopWatchInterface *cannyCVTimer = nullptr;

void cannyCV(int image_number)
{
    switch(image_number)
    {
        case 1:
            canny_cv_input_file = "Images\\Input\\img1.jpg";
            canny_cv_output_file = "Images\\Output\\Canny-CV-1.jpg";
        break;
        case 2:
            canny_cv_input_file = "Images\\Input\\img2.jpg";
            canny_cv_output_file = "Images\\Output\\Canny-CV-2.jpg";
        break;
        case 3:
            canny_cv_input_file = "Images\\Input\\img3.jpg";
            canny_cv_output_file = "Images\\Output\\Canny-CV-3.jpg";
        break;
        case 4:
            canny_cv_input_file = "Images\\Input\\img4.jpg";
            canny_cv_output_file = "Images\\Output\\Canny-CV-4.jpg";
        break;
        case 5:
            canny_cv_input_file = "Images\\Input\\img5.jpg";
            canny_cv_output_file = "Images\\Output\\Canny-CV-5.jpg";
        break;
        default:
            std::cerr << std::endl << "Error ... Please Enter Valid Number ... Exiting !!!" << std::endl;
            cvSobelCleanup();
            exit(EXIT_FAILURE);
        break;
    }

    sdkCreateTimer(&cannyCVTimer);
    
    input_image = cv::imread(canny_cv_input_file, cv::IMREAD_GRAYSCALE);

    sdkStartTimer(&cannyCVTimer);
    cv::GaussianBlur(input_image, output_image, cv::Size(3, 3), 0);
    cv::Canny(output_image, output_image, THRESHOLD, THRESHOLD * RATIO, 3);
    sdkStopTimer(&cannyCVTimer);

    std::cout << std::endl << "Time for Canny Operator using OpenCV (CPU) : " << sdkGetTimerValue(&cannyCVTimer) << " ms" << std::endl;

    cv::imwrite(canny_cv_output_file, output_image);

    cvSobelCleanup();
}

void cvSobelCleanup(void)
{
    if (cannyCVTimer)
    {
        sdkDeleteTimer(&cannyCVTimer);
        cannyCVTimer = nullptr;
    }

    output_image.release();
    input_image.release();
}
