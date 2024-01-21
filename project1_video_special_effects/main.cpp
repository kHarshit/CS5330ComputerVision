#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "include/imgDisplay.h"
#include "include/vidDisplay.h"

using namespace cv;

int main()
{
    // displayImage("/Users/harshit/Documents/CS5330ComputerVision/test_app/starry_night.jpg");
    displayVideo();
    return 0;
}