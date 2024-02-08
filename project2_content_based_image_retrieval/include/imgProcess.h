/**
 * author: Harshit Kumar, Khushi Neema
 * date: Jan 29, 2024
 * purpose: Implements various image processing features wrt Content-based Image Retrieval
 * such as baseline matching, histogram matching, etc.
 * 
*/

#ifndef IMG_PROCESS_H
#define IMG_PROCESS_H

#include <opencv2/opencv.hpp>


/**
 * @brief Compute features of an image
 * 
 * @param image Input image
 * @return cv::Mat Feature vector
*/
cv::Mat computeBaselineFeatures(const cv::Mat& image);

/**
 * @brief Compute distance between two feature vectors
 * 
 * @param features1 Feature vector 1
 * @param features2 Feature vector 2
 * @return double Distance between the two feature vectors
*/
double computeDistance(const std::vector<float>& feature1, const std::vector<float>& feature2);

/**
 * @brief Compute 2D RG Chromaticity Histogram
 * 
 * @param image Input image
 * @param bins Number of bins for the histogram
 * @return cv::Mat 2D RG Chromaticity Histogram
*/
cv::Mat computeRGChromaticityHistogram(const cv::Mat& image, int bins);

/**
 * @brief Compute histogram intersection between two histograms
 * 
 * @param hist1 Histogram 1
 * @param hist2 Histogram 2
 * @return float Histogram intersection value
*/
float histogramIntersection(const cv::Mat& hist1, const cv::Mat& hist2);
#endif // IMG_PROCESS_H
