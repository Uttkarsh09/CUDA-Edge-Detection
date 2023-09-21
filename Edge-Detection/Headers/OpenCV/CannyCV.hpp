#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

#ifndef HELPER_TIMER_H
#define HELPER_TIMER_H
    #include "../helper_timer.h"
#endif

#ifndef NOMINMAX
    #define NOMINMAX
#endif

#define THRESHOLD   40
#define RATIO       3

void cannyCV(int);
void cvSobelCleanup(void);
