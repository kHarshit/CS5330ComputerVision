/**
 * @author Harshit Kumar, Khushi Neema
 * @file distance.h
 * @brief This file provides the declarations for various distance metrics.
 */

#ifndef DISTANCE_METRICS_H
#define DISTANCE_METRICS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

/**
 * @brief Compute sum of squared distance between two feature vectors
 *
 * @param features1 Feature vector 1
 * @param features2 Feature vector 2
 * @return double SSD between the two feature vectors
 */
double sumSquaredDistance(const std::vector<float> &feature1, const std::vector<float> &feature2);

/**
 * @brief Compute cosine distance between two feature vectors
 *
 * @param feature1 Feature vector 1
 * @param feature2 Feature vector 2
 * @return double Cosine distance between the two feature vectors
 */
double cosineDistance(const std::vector<float> &feature1, const std::vector<float> &feature2);

/**
 * @brief Compute histogram intersection between two 2D histograms
 *
 * @param hist1 Histogram 1
 * @param hist2 Histogram 2
 * @return float Histogram intersection value
 */
double histogramIntersection2d(const cv::Mat &hist1, const cv::Mat &hist2);

/**
 * @brief Compute histogram intersection between two 3D histograms
 *
 * @param hist1 3D Histogram 1
 * @param hist2 3D Histogram 2
 * @return double Histogram intersection value
 */
double histogramIntersection3d(const cv::Mat &hist1, const cv::Mat &hist2);

/**
 * @brief Compute combined textured histogram distance
 *
 * @param histPair1 Pair of spatial histograms 1
 * @param histPair2 Pair of spatial histograms 2
 * @return double Combined histogram distance
 */
double combinedHistogramDistance_texture(const std::pair<cv::Mat, cv::Mat> &histPair1, const std::pair<cv::Mat, cv::Mat> &histPair2);

/**
 * @brief Compute combined histogram distance
 *
 * @param histPair1 Pair of spatial histograms 1
 * @param histPair2 Pair of spatial histograms 2
 * @return double Combined histogram distance
 */
double combinedHistogramDistance(const std::pair<cv::Mat, cv::Mat> &histPair1, const std::pair<cv::Mat, cv::Mat> &histPair2);

/**
 * @brief Compute composite weighted distance between two images using various features
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
double compositeDistance(const cv::Mat &hist1, const cv::Mat &hist2, double edgeDensity1, double edgeDensity2, double grassCoverage1, double grassCoverage2, const std::vector<float> &dnnFeatures1, const std::vector<float> &dnnFeatures2);

/**
 * @brief Compute composite weighted distance between two images using various features
 * @param hist1 RGB histogram 1
 * @param hist2 RGB histogram 2
 * @param dnnFeatures1 DNN features 1
 * @param dnnFeatures2 DNN features 2
 * @param weightHist Weight for histogram
 * @param weightDNN Weight for DNN features
 * @return double Distance
 */
double compositeDistanceBins(const cv::Mat &hist1, const cv::Mat &hist2, const std::vector<float> &dnnFeatures1, const std::vector<float> &dnnFeatures2, double weightHist, double weightDNN);

#endif // DISTANCE_METRICS_H