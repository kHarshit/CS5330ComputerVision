/**
 * author: Harshit Kumar, Khushi Neema
 * date: Jan 29, 2024
 * purpose: Implements various image processing features wrt Content-based Image Retrieval
 * such as baseline matching, histogram matching, etc.
 * 
*/

#include "imgProcess.h"

// cv::Mat computeBaselineFeatures(const cv::Mat& image) {
//     // Implementation to compute features (e.g., 7x7 square in the middle)
//     return image(cv::Rect(image.cols / 4, image.rows / 4, image.cols / 2, image.rows / 2)).clone();
// }

cv::Mat computeBaselineFeatures(const cv::Mat& image) {
    // Ensure the image is large enough for a 7x7 extraction
    if (image.cols < 7 || image.rows < 7) {
        throw std::runtime_error("Image is too small for a 7x7 feature extraction.");
    }
    int startX = (image.cols - 7) / 2;
    int startY = (image.rows - 7) / 2;
    return image(cv::Rect(startX, startY, 7, 7)).clone();
}


double computeDistance(const std::vector<float>& feature1, const std::vector<float>& feature2) {
    if (feature1.size() != feature2.size()) {
        std::cerr << "Error: Feature vectors must have the same length!" << std::endl;
        return -1.0; // Changed from -1.0f to -1.0 to match the return type
    }

    double sumSquaredDifferences = 0.0; // Removed the 'f' to ensure it's a double
    for (size_t i = 0; i < feature1.size(); ++i) {
        double diff = static_cast<double>(feature1[i]) - static_cast<double>(feature2[i]);
        sumSquaredDifferences += diff * diff;
    }

    return sumSquaredDifferences;
}

// Function to compute 2D RG Chromaticity Histogram
cv::Mat computeRGChromaticityHistogram(const cv::Mat& image, int bins) {
    cv::Mat histogram = cv::Mat::zeros(bins, bins, CV_32F);

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            float R = pixel[2];
            float G = pixel[1];
            float B = pixel[0];
            float sum = R + G + B;

            if (sum > 0) { // Avoid division by zero
                float r = R / sum;
                float g = G / sum;

                int r_bin = std::min(static_cast<int>(r * bins), bins - 1);
                int g_bin = std::min(static_cast<int>(g * bins), bins - 1);

                histogram.at<float>(r_bin, g_bin) += 1.0f;
            }
        }
    }

    // Normalize the histogram so that the sum of histogram bins = 1
    cv::normalize(histogram, histogram, 1, 0, cv::NORM_L1);
    return histogram;
}

float histogramIntersection(const cv::Mat& hist1, const cv::Mat& hist2) {
    CV_Assert(hist1.size() == hist2.size() && hist1.type() == hist2.type());

    float intersection = 0.0f;
    for (int r = 0; r < hist1.rows; ++r) {
        for (int g = 0; g < hist1.cols; ++g) {
            intersection += std::min(hist1.at<float>(r, g), hist2.at<float>(r, g));
        }
    }

    return intersection;
}
