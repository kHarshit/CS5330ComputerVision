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
#include<opencv2/core/core.hpp>


/**
 * @brief Compute features of an image
 * 
 * @param image Input image
 * @return cv::Mat Feature vector
*/
std::vector <float> computeBaselineFeatures(const cv::Mat& image);

/**
 * @brief Compute sum of squared distance between two feature vectors
 * 
 * @param features1 Feature vector 1
 * @param features2 Feature vector 2
 * @return double SSD between the two feature vectors
*/
double sumSquaredDistance(const std::vector<float>& feature1, const std::vector<float>& feature2);

/**
 * @brief Compute cosine distance between two feature vectors
 * 
 * @param feature1 Feature vector 1
 * @param feature2 Feature vector 2
 * @return double Cosine distance between the two feature vectors
*/
double cosineDistance(const std::vector<float> &feature1, const std::vector<float> &feature2);

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
double histogramIntersection(const cv::Mat& hist1, const cv::Mat& hist2);

/**
 * @brief Compute histogram intersection between two 3D histograms
 * 
 * @param hist1 3D Histogram 1
 * @param hist2 3D Histogram 2
 * @return double Histogram intersection value
*/
double histogramIntersection3d(const cv::Mat& hist1, const cv::Mat& hist2);

/**
 * @brief Compute RGB histogram
 * 
 * @param imagePart Input image
 * @param bins Number of bins for the histogram
 * @return cv::Mat RGB histogram
*/
cv::Mat computeRGBHistogram(const cv::Mat& imagePart, int bins);

/**
 * @brief Compute spatial histograms
 * 
 * @param image Input image
 * @param bins Number of bins for the histogram
 * @return std::pair<cv::Mat, cv::Mat> Pair of spatial histograms
*/
std::pair<cv::Mat, cv::Mat> computeSpatialHistograms(const cv::Mat& image, int bins);

/**
 * @brief Compute combined histogram distance
 * 
 * @param histPair1 Pair of spatial histograms 1
 * @param histPair2 Pair of spatial histograms 2
 * @return double Combined histogram distance
*/
double combinedHistogramDistance(const std::pair<cv::Mat, cv::Mat>& histPair1, const std::pair<cv::Mat, cv::Mat>& histPair2);

/**
 * @brief Compute grass chromaticity histogram
 * 
 * @param image Input image
 * @param bins Number of bins for the histogram
 * @return cv::Mat Grass chromaticity histogram
*/
cv::Mat computeGrassChromaticityHistogram(const cv::Mat &image, int bins);

/**
 * @brief Compute edge density of an image
 * 
 * @param image Input image
 * @return double Edge density
*/
double computeEdgeDensity(const cv::Mat& image);

/**
 * @brief Compute grass coverage in an image
 * 
 * @param image Input image
 * @return double Grass coverage
*/
double computeGrassCoverage(const cv::Mat& image);

/**
 * @brief Compute composite distance between two images using various features
 * 
 * @param hist1 RGB histogram 1
 * @param hist2 RGB histogram 2
 * @param edgeDensity1 Edge density 1
 * @param edgeDensity2 Edge density 2
 * @param grassCoverage1 Grass coverage 1
 * @param grassCoverage2 Grass coverage 2
 * @param dnnFeatures1 DNN features 1
 * @param dnnFeatures2 DNN features 2
 * @return double Composite distance
*/
double compositeDistance(const cv::Mat& hist1, const cv::Mat& hist2, double edgeDensity1, double edgeDensity2, double grassCoverage1, double grassCoverage2, const std::vector<float>& dnnFeatures1, const std::vector<float>& dnnFeatures2);

#endif // IMG_PROCESS_H
