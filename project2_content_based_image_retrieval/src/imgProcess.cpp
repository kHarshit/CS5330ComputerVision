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

std::vector<float> matToVector(const cv::Mat &m) {
    //cv::Mat flat = m.reshape(1, m.total() * m.channels());
    cv::Mat flat;
    if (m.isContinuous()) {
        flat = m.reshape(1, m.total() * m.channels());
    } else {
        cv::Mat continuousM;
        m.copyTo(continuousM);
        flat = continuousM.reshape(1, continuousM.total() * continuousM.channels());
    }
    flat.convertTo(flat, CV_32F);
    return m.isContinuous() ? flat : flat.clone();
}

std::vector <float> computeBaselineFeatures(const cv::Mat& image) {
    // Ensure the image is large enough for a 7x7 extraction
    if (image.cols < 7 || image.rows < 7) {
        throw std::runtime_error("Image is too small for a 7x7 feature extraction.");
    }
    int startX = (image.cols - 7) / 2;
    int startY = (image.rows - 7) / 2;
    return matToVector(image(cv::Rect(startX, startY, 7, 7)));
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
