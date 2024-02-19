/**
 * author: Harshit Kumar, Khushi Neema
 * date: Feb 19, 2024
 * purpose: Implements various object detection algorithms
 *
 */

#ifndef OBJ_DETECT_H
#define OBJ_DETECT_H

#include <opencv2/opencv.hpp>

double calculateDynamicThreshold(const cv::Mat& src, int k);

cv::Mat customThreshold(const cv::Mat& grayImage, double thresh, double maxValue);

cv::Mat preprocessAndThreshold(const cv::Mat& frame);


#endif // OBJ_DETECT_H
