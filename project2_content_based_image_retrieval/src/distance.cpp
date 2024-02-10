#include "distance.h"
#include <iostream>

double sumSquaredDistance(const std::vector<float> &feature1, const std::vector<float> &feature2)
{
    if (feature1.size() != feature2.size())
    {
        std::cerr << "Error: Feature vectors must have the same length!" << std::endl;
        return -1.0;
    }

    double sumSquaredDifferences = 0.0;
    for (size_t i = 0; i < feature1.size(); ++i)
    {
        double diff = static_cast<double>(feature1[i]) - static_cast<double>(feature2[i]);
        sumSquaredDifferences += diff * diff;
    }

    return sumSquaredDifferences;
}

double cosineDistance(const std::vector<float> &feature1, const std::vector<float> &feature2)
{
    if (feature1.size() != feature2.size())
    {
        std::cerr << "Error: Feature vectors must have the same length!" << std::endl;
        return -1.0;
    }

    double dotProduct = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (size_t i = 0; i < feature1.size(); ++i)
    {
        dotProduct += feature1[i] * feature2[i];
        norm1 += feature1[i] * feature1[i];
        norm2 += feature2[i] * feature2[i];
    }

    double cosineSimilarity = dotProduct / (std::sqrt(norm1) * std::sqrt(norm2));
    double cosineDistance = 1.0 - cosineSimilarity;

    return cosineDistance;
}

double histogramIntersection2d(const cv::Mat &hist1, const cv::Mat &hist2)
{
    CV_Assert(hist1.size() == hist2.size() && hist1.type() == hist2.type());

    double intersection = 0.0;
    for (int r = 0; r < hist1.rows; ++r)
    {
        for (int g = 0; g < hist1.cols; ++g)
        {
            // take the minimum value of the two histograms at each bin
            intersection += std::min(hist1.at<float>(r, g), hist2.at<float>(r, g));
        }
    }

    return intersection;
}

double histogramIntersection3d(const cv::Mat &hist1, const cv::Mat &hist2)
{
    CV_Assert(hist1.size == hist2.size && hist1.type() == hist2.type());

    double intersection = 0.0;
    // Assuming hist1 and hist2 are CV_32F type
    for (int i = 0; i < hist1.size[0]; ++i)
    {
        for (int j = 0; j < hist1.size[1]; ++j)
        {
            for (int k = 0; k < hist1.size[2]; ++k)
            {
                // take the minimum value of the two histograms at each bin
                int idx[3] = {i, j, k};
                intersection += std::min(hist1.at<float>(idx), hist2.at<float>(idx));
            }
        }
    }
    // Convert intersection to a measure of distance
    return intersection;
}

double combinedHistogramDistance(const std::pair<cv::Mat, cv::Mat> &histPair1, const std::pair<cv::Mat, cv::Mat> &histPair2)
{
    // Compute histogram intersections
    double topIntersection = histogramIntersection3d(histPair1.first, histPair2.first);
    double bottomIntersection = histogramIntersection3d(histPair1.second, histPair2.second);
    std::cout << "topIntersection: " << topIntersection << " bottomIntersection: " << bottomIntersection << std::endl;

    // weighted average of distances
    // Assuming equal importance for top and bottom histograms
    double combinedDistance = 0.5 * topIntersection + 0.5 * bottomIntersection;

    // Convert intersection to a measure of distance
    return 1.0 - combinedDistance;
}

double compositeDistance(const cv::Mat &hist1, const cv::Mat &hist2, double edgeDensity1, double edgeDensity2, double grassCoverage1, double grassCoverage2, const std::vector<float> &dnnFeatures1, const std::vector<float> &dnnFeatures2)
{
    double colorDist = cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA);
    double edgeDist = std::abs(edgeDensity1 - edgeDensity2);
    double spatialDistance = std::abs(grassCoverage1 - grassCoverage2);
    double dnnDist = cosineDistance(dnnFeatures1, dnnFeatures2);

    // Weighted average of distances
    return colorDist * 0.5 + edgeDist * 0.1 + spatialDistance * 0.1 + dnnDist * 0.3;
}

double compositeDistanceBins(const cv::Mat &hist1, const cv::Mat &hist2, const std::vector<float> &dnnFeatures1, const std::vector<float> &dnnFeatures2, double weightHist, double weightDNN)
{
    double histDist = cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA);
    double dnnDist = cosineDistance(dnnFeatures1, dnnFeatures2);

    // Calculate weighted average
    double totalWeight = weightHist + weightDNN;
    double weightedAvg = (weightHist * histDist + weightDNN * dnnDist) / totalWeight;

    return weightedAvg;
}

double combinedHistogramDistance_texture(const std::pair<cv::Mat, cv::Mat> &histPair1, const std::pair<cv::Mat, cv::Mat> &histPair2)
{
    // Compute histogram intersections
    double topIntersection = histogramIntersection3d(histPair1.first, histPair2.first);
    double bottomIntersection = histogramIntersection2d(histPair1.second, histPair2.second);
    std::cout << "topIntersection: " << topIntersection << " bottomIntersection: " << bottomIntersection << std::endl;

    // weighted average of distances
    // Assuming equal importance for top and bottom histograms
    double combinedDistance = 0.5 * topIntersection + 0.5 * bottomIntersection;

    // Convert intersection to a measure of distance
    return 1.0 - combinedDistance;
}