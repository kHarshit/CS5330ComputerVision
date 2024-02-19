/**
 * author: Harshit Kumar, Khushi Neema
 * date: Feb 19, 2024
 * purpose: Implements various object detection algorithms
 *
 */

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include "objDetect.h"

using namespace std;
using namespace cv;

// Function to dynamically calculate the threshold using k-means (ISODATA algorithm)
double calculateDynamicThreshold(const Mat& src, int k) {
    Mat samples(src.rows * src.cols, 1, CV_32F);
    for(int y = 0; y < src.rows; y++)
        for(int x = 0; x < src.cols; x++)
            samples.at<float>(y + x * src.rows, 0) = src.at<uchar>(y, x);

    Mat labels, centers;
    kmeans(samples, k, labels, TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    // Assuming K=2, calculate the mean threshold
    double thresholdValue = (centers.at<float>(0, 0) + centers.at<float>(1, 0)) / 2.0;
    return thresholdValue;
}

cv::Mat customThreshold(const cv::Mat& grayImage, double thresh, double maxValue)
{
    // clone the input image
    cv::Mat outputImage = grayImage.clone();

    // loop over the input image and apply the thresholding
    for(int i = 0; i < grayImage.rows; ++i)
    {
        for(int j = 0; j < grayImage.cols; ++j)
        {
            if(grayImage.at<uchar>(i, j) < thresh)
            {
                outputImage.at<uchar>(i, j) = static_cast<uchar>(maxValue);
            }
            else
            {
                outputImage.at<uchar>(i, j) = 0;
            }
        }
    }

    return outputImage;
}

// Function to preprocess and threshold the video frame
Mat preprocessAndThreshold(const Mat& frame) {
    // Convert to grayscale
    Mat grayFrame;
    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

    // Optional: Blur the image to make regions more uniform
    GaussianBlur(grayFrame, grayFrame, Size(5, 5), 0);

    // Dynamically calculate the threshold
    double thresholdValue = calculateDynamicThreshold(grayFrame, 2);
    cout << "Threshold: " << thresholdValue << endl;

    // Apply the threshold
    // Mat thresholded;
    // threshold(grayFrame, thresholded, thresholdValue, 255, THRESH_BINARY_INV);
    cv::Mat thresholded = customThreshold(grayFrame, thresholdValue, 255);

    return thresholded;
}