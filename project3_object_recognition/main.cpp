#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "objDetect.h"

using namespace cv;

int main()
{
    // to open a default camera
    cv::VideoCapture cap(0);
    cv::Mat frame, blur, hsv;
    if (!cap.isOpened())
    {
        printf("Error opening the default camera");
        return -1;
    }
    cv::namedWindow("Original Video", WINDOW_NORMAL);
    cv::namedWindow("Thresholded", WINDOW_NORMAL);
    for (;;)
    {

        // cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            std::cout << "Error: Blank frame grabbed" << std::endl;
            continue;
        }
        cv::imshow("Original Video", frame);

        // Preprocess and threshold the frame
        cv::Mat thresholdedFrame = preprocessAndThreshold(frame);

#if 0
        cv::imshow("Blurred Video",hsv);
        
        blur5x5_2(frame,blur);
        
        //cv::cvtColor(frame,frame,cv::COLOR_BGR2GRAY);
        cv::convertScaleAbs(blur, blur);
    
        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // // Dynamically set the threshold using ISODATA algorithm
        std::vector<Mat> channels;
        split(hsv, channels);
        Mat sat = channels[1]; // Saturation channel
        double minVal, maxVal;
        minMaxLoc(sat, &minVal, &maxVal);
        double thresholdValue = (minVal + maxVal) / 2.0;

        // // Thresholding
        Mat thresholdedFrame;
        cv::threshold(sat, thresholdedFrame, thresholdValue, 255, THRESH_BINARY);
#endif

        // thresholded video
        cv::imshow("Thresholded", thresholdedFrame);

        char key = cv::waitKey(10);
        if (key == 'q')
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}