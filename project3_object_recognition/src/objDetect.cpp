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

int blur5x5_2(cv::Mat &src, cv::Mat &dst)
{
    dst = cv::Mat::zeros(src.size(), src.type());
    cv::Mat temp = cv::Mat::zeros(src.size(), src.type());

    // row filter
    for (int i = 2; i < src.rows - 2; i++)
    {
        cv::Vec3b *rowptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *tempptr = temp.ptr<cv::Vec3b>(i);

        // loop over columns
        for (int j = 2; j < src.cols - 2; j++)
        {
            // loop over color channels
            for (int c = 0; c < 3; c++)
            {
                // row filter [1 , 2, 4, 2, 1]
                tempptr[j][c] = (1 * rowptr[j - 2][c] + 2 * rowptr[j - 1][c] + 4 * rowptr[j][c] + 2 * rowptr[j + 1][c] + 1 * rowptr[j + 2][c]) / 10.0;
            }
        }
    }

    // column filter
    for (int i = 2; i < src.rows - 2; i++)
    {
        cv::Vec3b *rowptrm2 = temp.ptr<cv::Vec3b>(i - 2);
        cv::Vec3b *rowptrm1 = temp.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b *rowptr = temp.ptr<cv::Vec3b>(i);
        cv::Vec3b *rowptrp1 = temp.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b *rowptrp2 = temp.ptr<cv::Vec3b>(i + 2);

        cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);

        // loop over columns
        for (int j = 2; j < src.cols - 2; j++)
        {
            // loop over color channels
            for (int c = 0; c < 3; c++)
            {
                /*column filter
                    [1]  m2
                    [2]  m1
                    [4]  r
                    [2]  p1
                    [1]  p2
                */
                dptr[j][c] = (1 * rowptrm2[j][c] + 2 * rowptrm1[j][c] + 4 * rowptr[j][c] + 2 * rowptrp1[j][c] + 1 * rowptrp2[j][c]) / 10.0;
                // clip b/w 0 and 255
                dptr[j][c] = dptr[j][c] > 255 ? 255 : dptr[j][c];
            }
        }
    }

    return 0; // Success
}

double calculateDynamicThreshold(const Mat &src, int k)
{
    // Reshape the image to a 1D array of pixels
    Mat samples(src.rows * src.cols, 1, CV_32F);
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
            samples.at<float>(y + x * src.rows, 0) = src.at<uchar>(y, x);

    // Apply k-means clustering
    Mat labels, centers;
    kmeans(samples, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    // Assuming K=2, calculate the mean threshold
    double thresholdValue = (centers.at<float>(0, 0) + centers.at<float>(1, 0)) / 2.0;
    return thresholdValue;
}

cv::Mat customThreshold(const cv::Mat &grayImage, double thresh, double maxValue)
{
    // clone the input image
    cv::Mat outputImage = grayImage.clone();

    // loop over the input image and apply the thresholding
    for (int i = 0; i < grayImage.rows; ++i)
    {
        for (int j = 0; j < grayImage.cols; ++j)
        {
            // apply the thresholding
            if (grayImage.at<uchar>(i, j) < thresh)
            {
                // set the pixel value to the maximum value
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
Mat preprocessAndThreshold(const cv::Mat &frame)
{
    // Convert to grayscale
    Mat grayFrame,blur;
    Mat input=frame;
    blur5x5_2(input, blur);
    cv::convertScaleAbs(blur,blur);
    cvtColor(blur, grayFrame, COLOR_BGR2GRAY);

    // Optional: Blur the image to make regions more uniform
    // blur5x5_2(grayFrame, grayFrame);
    // cv::convertScaleAbs(grayFrame,grayFrame);
    //GaussianBlur(grayFrame, grayFrame, Size(5, 5), 0);

    // Dynamically calculate the threshold
    double thresholdValue = calculateDynamicThreshold(grayFrame, 2);
    std::cout << "Threshold: " << thresholdValue << std::endl;

    // Apply the threshold
    // Mat thresholded;
    // threshold(grayFrame, thresholded, thresholdValue, 255, THRESH_BINARY_INV);
    cv::Mat thresholded = customThreshold(grayFrame, thresholdValue, 255);

    return thresholded;
}

void morphologyEx(const cv::Mat& src, cv::Mat& dst, int operation, const cv::Mat& kernel) {
    switch (operation) {
        case MORPH_DILATE:
            dilate(src, dst, kernel);
            break;
        case MORPH_ERODE:
            erode(src, dst, kernel);
            break;
        case MORPH_OPEN: {
            Mat temp;
            erode(src, temp, kernel);
            dilate(temp, dst, kernel);
            break;
        }
        case MORPH_CLOSE: {
            Mat temp;
            dilate(src, temp, kernel);
            erode(temp, dst, kernel);
            break;
        }
        default:
            std::cout<< "Invalid morphological operation" << std::endl;
            break;
    }
}