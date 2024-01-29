/**
 * author: Harshit Kumar, Khushi Neema
 * date: Jan 29, 2024
 * purpose: Implements various image processing features wrt Content-based Image Retrieval
 * such as baseline matching, histogram matching, etc.
 * 
*/

#include "imgProcess.h"

cv::Mat computeBaselineFeatures(const cv::Mat& image) {
    // Implementation to compute features (e.g., 7x7 square in the middle)
    return image(cv::Rect(image.cols / 4, image.rows / 4, image.cols / 2, image.rows / 2)).clone();
}

double computeDistance(const cv::Mat& features1, const cv::Mat& features2) {
    // Implementation to compute distance (e.g., sum-of-squared-difference)
    cv::Mat diff;
    cv::absdiff(features1, features2, diff);
    return cv::sum(diff)[0]; // Assuming single-channel images
}
