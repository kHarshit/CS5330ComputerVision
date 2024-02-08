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
                
                int r_bin = static_cast<int>(r * (bins - 1) + 0.5);
                int g_bin = static_cast<int>(g * (bins - 1) + 0.5);

                // int r_bin = std::min(static_cast<int>(r * bins), bins - 1);
                // int g_bin = std::min(static_cast<int>(g * bins), bins - 1);

                histogram.at<float>(r_bin, g_bin) += 1.0f;
            }
        }
    }

    // Normalize the histogram so that the sum of histogram bins = 1
    cv::normalize(histogram, histogram, 1, 0, cv::NORM_L1);
    return histogram;
}

double histogramIntersection(const cv::Mat& hist1, const cv::Mat& hist2) {
    CV_Assert(hist1.size() == hist2.size() && hist1.type() == hist2.type());

    double intersection = 0.0;
    for (int r = 0; r < hist1.rows; ++r) {
        for (int g = 0; g < hist1.cols; ++g) {
            intersection += std::min(hist1.at<float>(r, g), hist2.at<float>(r, g));
        }
    }

    return intersection;
}


cv::Mat computeRGBHistogram(const cv::Mat& image, int bins) {
    // Initialize a 3D histogram with given bins for each dimension and float type
    // Define the size for each dimension of the histogram
    int histSize[] = {bins, bins, bins};
    // Create a 3D histogram with floating-point values
    cv::Mat histogram(3, histSize, CV_32F, cv::Scalar(0));

    // cv::Mat histogram = cv::Mat::zeros(bins, bins, bins, CV_32F);

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            float R = pixel[2];
            float G = pixel[1];
            float B = pixel[0];

            // Calculate bin indices for R, G, and B
            int r_bin = static_cast<int>(R * (bins - 1) / 255.0 + 0.5);
            int g_bin = static_cast<int>(G * (bins - 1) / 255.0 + 0.5);
            int b_bin = static_cast<int>(B * (bins - 1) / 255.0 + 0.5);

            // Increment the corresponding bin
            // Ensure 3D access to histogram is correct: histogram.at<float>(b_bin, g_bin, r_bin) for CV_32F
            *histogram.ptr<float>(r_bin, g_bin, b_bin) += 1.0f;
        }
    }

    // Compute total number of pixels to normalize histogram
    float totalPixels = image.rows * image.cols;

    // Normalize the histogram so that the sum of histogram bins = 1
    histogram /= totalPixels;

    return histogram;
}

std::pair<cv::Mat, cv::Mat> computeSpatialHistograms(const cv::Mat& image, int bins) {
    // Split image into top and bottom halves
    cv::Mat topHalf = image(cv::Rect(0, 0, image.cols, image.rows / 2));
    cv::Mat bottomHalf = image(cv::Rect(0, image.rows / 2, image.cols, image.rows / 2));

    // Compute RGB histograms for each part
    cv::Mat topHist = computeRGBHistogram(topHalf, bins);
    cv::Mat bottomHist = computeRGBHistogram(bottomHalf, bins);

    return {topHist, bottomHist};
}

double combinedHistogramDistance(const std::pair<cv::Mat, cv::Mat>& histPair1, const std::pair<cv::Mat, cv::Mat>& histPair2) {
    // Compute histogram intersections
    float topIntersection = histogramIntersection(histPair1.first, histPair2.first);
    float bottomIntersection = histogramIntersection(histPair1.second, histPair2.second);

    // Example of a simple weighted average of distances
    // Assuming equal importance for top and bottom histograms
    double combinedDistance = (topIntersection + bottomIntersection) / 2.0;

    // Convert intersection to a measure of distance
    return 1.0 - combinedDistance;
}

