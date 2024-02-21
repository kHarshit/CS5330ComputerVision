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
 * @brief Apply 5x5 Gaussian blur to the input image
 *
 * @param src input image
 * @param dst output image
 * @return int
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Calculate the dynamic threshold using k-means clustering algorithm
 *
 * @param src input image
 * @param k number of clusters
 * @return double
 */
double calculateDynamicThreshold(const cv::Mat &src, int k);

/**
 * @brief Apply custom thresholding to the input image
 *
 * @param grayImage input image
 * @param thresh threshold value
 * @param maxValue maximum value
 * @return cv::Mat
 */
cv::Mat customThreshold(const cv::Mat &grayImage, double thresh, double maxValue);

/**
 * @brief Preprocess and threshold the video frame
 *
 * @param frame input frame
 * @return cv::Mat
 */
cv::Mat preprocessAndThreshold(const cv::Mat &frame);
/**
* @brief Cleans noise/holes from the image
*
*
*@param src Input image frame
*@param dst Applied cleaned filtered image
*@param operation deciding what operation is needed to be performed (growing, shrinking, growing + shrinking, shrinking + growing)
*/
void morphologyEx(const cv::Mat& src, cv::Mat& dst, int operation, const cv::Mat& kernel) ;

/**
* @brief Assigns labels to the binary image by utilizing DFS method
*
*
*@param binaryImage Input image frame
*@param labels storing labels for each pixel
*@param i x-coordinae of the image
*@param j y-coordinae of the image
*@param label current label to be assigned
*/
void dfs(const cv::Mat& binaryImage, cv::Mat& labels, int i, int j, int label);
/**
* @brief Assigns labels to the binary image 
*
*
*@param binaryImage Input image frame
*@param labeledImage getting labels for every pixels of an image
*/
void connectedComponents(const cv::Mat& binaryImage, cv::Mat& labeledImage);

#endif // OBJ_DETECT_H
