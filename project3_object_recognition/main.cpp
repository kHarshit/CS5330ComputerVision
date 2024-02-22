#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "objDetect.h"

using namespace cv;
using namespace std;


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
    cv::namedWindow("Cleaned Threholded",WINDOW_NORMAL);
    cv::namedWindow("Conected Components",WINDOW_NORMAL);
    for (;;)
    {

        // cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            std::cout << "Error: Blank frame grabbed" << std::endl;
            continue;
        }
        

        // 1. Preprocess and threshold the frame
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


        // 2. Clean the thresholded frame
        cv::Mat cleaned;
        // use 11x11 kernel
        // Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11));
        // use 3x3 8-connected kernel
        cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
        // or use 3x3 4-connected kernel
        // 0 1 0
        // 1 1 1
        // 0 1 0
        // cv::Mat kernel = cv::Mat::zeros(3, 3, CV_8U);
        // kernel.at<uchar>(1, 0) = 1;
        // kernel.at<uchar>(0, 1) = 1;
        // kernel.at<uchar>(1, 1) = 1;
        // kernel.at<uchar>(1, 2) = 1;
        // kernel.at<uchar>(2, 1) = 1;

        morphologyEx(thresholdedFrame,cleaned,MORPH_CLOSE,kernel);
        
        cv:Mat labeledImage(cleaned.size(), CV_32S); 
        cv::Mat labeledImage8U;
        
        // 3. Find connected components
        cv::Mat colorLabeledImage;
        connectedComponentsTwoPass(cleaned,labeledImage);
        cv::normalize(labeledImage, labeledImage, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(labeledImage, colorLabeledImage, cv::COLORMAP_JET);

        // 4. Compute features for each connected component

        // Display the images
        cv::imshow("Original Video", frame);
        cv::imshow("Connected Components", colorLabeledImage);
        cv::imshow("Thresholded", thresholdedFrame);
        cv::imshow("Cleaned thresholded",cleaned);
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