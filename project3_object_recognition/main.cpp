#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <map>
#include <vector>
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
    cv::namedWindow("Connected Components Features",WINDOW_NORMAL);
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
        cv::Mat cleanedImg;
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

        morphologyEx(thresholdedFrame,cleanedImg,MORPH_CLOSE,kernel);
        
        cv::Mat labeledImg(cleanedImg.size(), CV_32S); 
        cv::Mat labeledImgNormalized;
        
        // 3. Find connected components
        cv::Mat colorLabeledImg;
        // returns a map of connected components {pixel number: connected component number}
        std::map<int, int> connectedComponents = connectedComponentsTwoPass(cleanedImg, labeledImg);
        cv::normalize(labeledImg, labeledImgNormalized, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(labeledImgNormalized, colorLabeledImg, cv::COLORMAP_JET);

        // 4. Compute features for each connected component
        cv::Mat colorLabeledFeatureImg;
        cv::Mat featureOutImg;
        computeFeatures(labeledImg, connectedComponents, featureOutImg);
        cv::normalize(featureOutImg, featureOutImg, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(featureOutImg, colorLabeledFeatureImg, cv::COLORMAP_JET);

        // Display the images
        cv::imshow("0. Original Video", frame);
        cv::imshow("1. Thresholded", thresholdedFrame);
        cv::imshow("2. Cleaned thresholded",cleanedImg);
        cv::imshow("3. Connected Components", colorLabeledImg);
        cv::imshow("4. Connected Components Features", colorLabeledFeatureImg);
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