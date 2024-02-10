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
#include <opencv2/core/core.hpp>

/**
 * @brief Compute features of an image
 *
 * @param image Input image
 * @return cv::Mat Feature vector
 */
std::vector<float> computeBaselineFeatures(const cv::Mat &image);

/**
 * @brief Compute 2D RG Chromaticity Histogram
 *
 * @param image Input image
 * @param bins Number of bins for the histogram
 * @return cv::Mat 2D RG Chromaticity Histogram
 */
cv::Mat computeRGChromaticityHistogram(const cv::Mat &image, int bins);

/**
 * @brief Compute RGB histogram
 *
 * @param imagePart Input image
 * @param bins Number of bins for the histogram
 * @return cv::Mat RGB histogram
 */
cv::Mat computeRGBHistogram(const cv::Mat &imagePart, int bins);

/**
 * @brief Compute spatial histograms
 *
 * @param image Input image
 * @param bins Number of bins for the histogram
 * @return std::pair<cv::Mat, cv::Mat> Pair of spatial histograms
 */
std::pair<cv::Mat, cv::Mat> computeSpatialHistograms(const cv::Mat &image, int bins);
/**
 * @brief Compute spatial textured histograms
 *
 * @param image Input image
 * @param bins Number of bins for the histogram
 * @return std::pair<cv::Mat, cv::Mat> Pair of spatial histograms
 */
std::pair<cv::Mat, cv::Mat> computeSpatialHistograms_texture(const cv::Mat &image, int bins);

/*
 * @brief Applies SobelX filter (horizontal edge detection) to an image
 * @param src Input image
 * @param dst Output image
 * @return 0 if the operation is succesful.
 */

int sobelX3x3(cv::Mat &src, cv::Mat &dst);

/*
 * @brief Applies SobelY filter (vertical edge detection) to an image
 * @param src Input image
 * @param dst Output image
 * @return 0 if the operation is succesful.
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

/**
 * @brief generates a gradient magnitude to an image.
 * @param sobelX Input image.
 * @param sobelY Input image.
 * @param dst Output image.
 * @return 0 if the operation is successful.
 */
int magnitude(cv::Mat &sobelX, cv::Mat &sobelY, cv::Mat &dst);

/**
 * @brief generates a gradient magnitude to an image.
 * @param image Input image.
 * @param bins Number of bins in a histogram.
 * @return cv::Mat gradient magnitude image.
 */
cv::Mat texture(const cv::Mat image, int bins);

/**
 * @brief generates an orientation to an image.
 * @param image Input image.
 * @param sx SobelX of an image.
 * @param sy SobelY of an image.
 * @return cv::Mat orientated image.
 */
cv::Mat orientation(cv::Mat &image, cv::Mat sx, cv::Mat sy);

/**
 * @brief Compute grass chromaticity histogram
 *
 * @param image Input image
 * @param bins Number of bins for the histogram
 * @return cv::Mat Grass chromaticity histogram
 */
cv::Mat computeGrassChromaticityHistogram(const cv::Mat &image, int bins);

/**
 * @brief Compute blue chromaticity histogram
 * @param image Input image
 * @param bins Number of bins for the histogram
 * @return cv::Mat Blue chromaticity histogram
 */
cv::Mat computeBlueChromaticityHistogram(const cv::Mat &image, int bins);

/**
 * @brief Compute edge density of an image
 *
 * @param image Input image
 * @return double Edge density
 */
double computeEdgeDensity(const cv::Mat &image);

/**
 * @brief Compute grass coverage in an image
 *
 * @param image Input image
 * @return double Grass coverage
 */
double computeGrassCoverage(const cv::Mat &image);

/**
 * @brief Creates textured image using gabor filter.
 * @param image input image
 * @param bins Number of bins in histogram.
 * @return float textured image vector
 */
cv::Mat gaborTexture(cv::Mat &image, int bins);

/**
 * @brief Compute spatial textured histograms with gabor filter
 *
 * @param image Input image
 * @param bins Number of bins for the histogram
 * @return std::pair<cv::Mat, cv::Mat> Pair of spatial histograms
 */
std::pair<cv::Mat, cv::Mat> computeSpatialHistograms_gabor(const cv::Mat &image, int bins);

#endif // IMG_PROCESS_H
