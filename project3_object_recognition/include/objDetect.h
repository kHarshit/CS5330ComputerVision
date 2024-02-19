/**
 * author: Harshit Kumar, Khushi Neema
 * date: Feb 19, 2024
 * purpose: Implements various object detection algorithms
 *
 */

#ifndef OBJ_DETECT_H
#define OBJ_DETECT_H

#include <opencv2/opencv.hpp>

/**
 * @brief Calculate the dynamic threshold using k-means clustering algorithm
 * 
 * @param src input image
 * @param k number of clusters
 * @return double 
*/
double calculateDynamicThreshold(const cv::Mat& src, int k);

/**
 * @brief Apply custom thresholding to the input image
 * 
 * @param grayImage input image
 * @param thresh threshold value
 * @param maxValue maximum value
 * @return cv::Mat
*/
cv::Mat customThreshold(const cv::Mat& grayImage, double thresh, double maxValue);

/**
 * @brief Preprocess and threshold the video frame
 * 
 * @param frame input frame
 * @return cv::Mat
*/
cv::Mat preprocessAndThreshold(const cv::Mat& frame);


#endif // OBJ_DETECT_H
