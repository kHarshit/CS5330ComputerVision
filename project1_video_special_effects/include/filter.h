/**
 * author: Harshit Kumar
 * date: Jan 20, 2024
 * purpose: Applies a filter to an image.
 * 
*/

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>

/**
 * @brief Applies a custom greyscale filter to an image.
 * @param src Input image.
 * @param dst Output image.
 * @return 0 if the operation is successful.
 * 
*/
int greyscale(cv::Mat& src, cv::Mat& dst);

/**
 * @brief Applies a custom sepia filter to an image.
 * @param src Input image.
 * @param dst Output image.
 * @return 0 if the operation is successful.
 * 
*/
int sepia(cv::Mat& src, cv::Mat& dst);

/**
 * @brief Applies a custom blur filter to an image.
 * @param src Input image.
 * @param dst Output image.
 * @return 0 if the operation is successful.
*/
int blur5x5_1( cv::Mat &src, cv::Mat &dst );

/**
 * @brief Applies a custom blur filter to an image (faster version).
 * @param src Input image.
 * @param dst Output image.
 * @return 0 if the operation is successful.
*/
int blur5x5_2( cv::Mat &src, cv::Mat &dst );

#endif // FILTER_H