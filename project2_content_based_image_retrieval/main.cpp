/**
 * author: Harshit Kumar, Khushi Neema
 * date: Jan 29, 2024
 */

#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <vector>
#include "include/imgProcess.h"
#include "include/csv_util.h"

using namespace std;

// Run Part 1: reads the target image and computes its features,
// loops over the directory of images, and for each image computes the features
// and compares them to the target image, storing the result in an array or vector,
// sorts the list of matches and returns the top N
// Part 2: writes the feature vector for each image to a file to save processing time
#define RUN_PART1 1
// if RUN_PART1 is 0 and you want to write data to csv, set it to 1
#define WRITE_CSV 0

void sort(std::vector<std::pair<std::string, double>> &distances, bool ascending = true)
{
    // Sorting the distances in ascending or descending order based on the 'ascending' argument
    if (ascending)
    {
        std::sort(distances.begin(), distances.end(), [](const std::pair<std::string, double> &a, const std::pair<std::string, double> &b)
                  { return a.second < b.second; });
    }
    else
    {
        std::sort(distances.begin(), distances.end(), [](const std::pair<std::string, double> &a, const std::pair<std::string, double> &b)
                  { return a.second > b.second; });
    }
}

cv::Rect selectROI(const cv::Mat& image) {
    // Let the user select a region of interest on the image
    cv::Rect roi = cv::selectROI("Select ROI", image);
    cv::destroyWindow("Select ROI");
    return roi;
}

#if !RUN_PART1

int main(int argc, char *argv[])
{

#if WRITE_CSV
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <directory path> <output CSV file>" << std::endl;
        return -1;
    }

    DIR *dirp;
    struct dirent *dp;
    dirp = opendir(argv[1]);
    if (dirp == NULL)
    {
        std::cout << "Cannot open directory " << argv[1] << std::endl;
        return -1;
    }

    bool reset_file = true; // Reset the file for the first image
    while ((dp = readdir(dirp)) != NULL)
    {
        if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif"))
        {
            std::string filepath = std::string(argv[1]) + "/" + std::string(dp->d_name);
            cv::Mat image = cv::imread(filepath);
            if (!image.empty())
            {
                std::vector<float> features = computeBaselineFeatures(image);

                // Convert features to a vector<float>
                // std::vector<float> feature_vector(features.begin<float>(), features.end<float>());
                append_image_data_csv(argv[2], const_cast<char *>(filepath.c_str()), features, reset_file);
                reset_file = false; // Only reset the file once, for the first image
            }
        }
    }

    closedir(dirp);
#else
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << " <target image> <feature vector file> <N>" << std::endl;
        return -1;
    }
    std::string imgPath = argv[1];
    std::string targetImageFilename = imgPath.substr(imgPath.find_last_of("/\\") + 1);
    cv::Mat target_image = cv::imread(argv[1]);
    if (target_image.empty())
    {
        std::cout << "Could not read the target image." << std::endl;
        return -1;
    }

    int N = std::atoi(argv[3]);

    // std::vector<float> target_feature_vector = computeBaselineFeatures(target_image);
    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;
    if (read_image_data_csv(argv[2], filenames, data, 0) != 0)
    {
        std::cerr << "Error reading feature vector file." << std::endl;
        return -1;
    }

    // Find the feature vector for the target image
    std::vector<float> target_feature_vector;
    bool targetFound = false;
    for (size_t i = 0; i < filenames.size(); ++i)
    {
        if (std::string(filenames[i]) == targetImageFilename)
        {
            target_feature_vector = data[i];
            targetFound = true;
            break;
        }
    }

    if (!targetFound)
    {
        std::cerr << "Feature vector for target image not found." << std::endl;
        return -1;
    }

    std::vector<std::pair<std::string, double>> distances;
    for (size_t i = 0; i < data.size(); ++i)
    {
        // double distance = sumSquaredDistance(target_feature_vector, data[i]);
        double distance = cosineDistance(target_feature_vector, data[i]);
        if (distance >= 0)
        { // Ensure distance is valid
            cout << "Distance: " << distance << endl;
            distances.push_back(std::make_pair(std::string(filenames[i]), distance));
        }
    }

    // Sorting the distances in ascending order
    sort(distances);

    // Displaying top N matches, starting from the second match to avoid the target image itself if present
    for (int i = 1; i <= N && i < distances.size(); ++i)
    {
        std::cout << "Distance: " << distances[i].second << ", File: " << distances[i].first << std::endl;
    }

#endif

    return 0;
}

#else
int main(int argc, char *argv[])
{
    char dirname[256];
    char buffer[256];
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;

    cv::Mat target_image, target_features;
    std::vector<float> target_feature_vector;
    std::pair<cv::Mat, cv::Mat> target_hist;
    if (argc < 4)
    {
        printf("usage: %s <directory path> <target image path> <feature type> <n> <dnn feature file> <select ROI boolean>\n", argv[0]);
        exit(-1);
    }

    strcpy(dirname, argv[1]);
    printf("Processing directory %s\n", dirname);
    string featureType = argv[3];
    cout << "Feature type: " << featureType << endl;
    // number of similar images to display
    int N = std::atoi(argv[4]);

    // Reading the target image and computing its features
    std::string imgPath = argv[2];
    std::string targetImageFilename = imgPath.substr(imgPath.find_last_of("/\\") + 1);
    target_image = cv::imread(argv[2]);
    cv::imshow("Target Image", target_image);
    cv::waitKey(0);

    // Allow the user to select a Region of Interest (ROI) from the target image
    bool selectROIbool = 0; // Default value
    if (argc > 6) {
        selectROIbool = std::atoi(argv[6]);
    }
    cv::Mat targetImgROI = target_image;
    if (selectROIbool)
    {
        cv::Rect roi = selectROI(target_image);
        if (roi.width > 10 && roi.height > 10)
        { // Check if a valid ROI was selected
            // Crop the target image to the selected ROI
            targetImgROI = target_image(roi);
        }
        else
        {
            cout << "No Valid ROI selected. Using the entire image." << endl;
        }
    }
    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;
    if (argc > 5)
    {
        if (read_image_data_csv(argv[5], filenames, data, 0) != 0)
        {
            std::cerr << "Error reading feature vector file." << std::endl;
            return -1;
        }
    }
    else
    {
        std::cout << "Feature vector file not provided. Some feature types may not work." << std::endl;
    }

    double edgeDensity = -1.0;
    double grassCoverage = -1.0;

    printf("Computing target features : ");
    if (featureType == "baseline")
    {
        target_feature_vector = computeBaselineFeatures(targetImgROI);
    }
    else if (featureType == "histogram")
    {
        target_features = computeRGChromaticityHistogram(targetImgROI, 16);
    }
    else if (featureType == "multihistogram")
    {
        target_hist = computeSpatialHistograms(targetImgROI, 8);
    }
    else if (featureType == "dnn")
    {
        // Find the feature vector for the target image
        bool targetFound = false;
        for (size_t i = 0; i < filenames.size(); ++i)
        {
            // std::cout<<std::string(filenames[i]);
            if (std::string(filenames[i]) == targetImageFilename)
            {

                target_feature_vector = data[i];
                targetFound = true;
                break;
            }
        }
        if (!targetFound)
        {
            std::cerr << "Feature vector for target image not found." << std::endl;
            return -1;
        }
    }
    else if (featureType == "grass")
    {
        // Find the feature vector for the target image
        bool targetFound = false;
        for (size_t i = 0; i < filenames.size(); ++i)
        {
            // std::cout<<std::string(filenames[i]);
            if (std::string(filenames[i]) == targetImageFilename)
            {

                target_feature_vector = data[i];
                targetFound = true;
                break;
            }
        }
        if (!targetFound)
        {
            std::cerr << "Feature vector for target image not found." << std::endl;
            return -1;
        }
        target_features = computeGrassChromaticityHistogram(targetImgROI, 16);
        edgeDensity = computeEdgeDensity(targetImgROI);
        grassCoverage = computeGrassCoverage(targetImgROI);
    }
    else if (featureType == "texture")
    {
        target_hist = computeSpatialHistograms_texture(targetImgROI, 8);
    }
    else if (featureType == "gabor")
    {
        target_hist = computeSpatialHistograms_gabor(targetImgROI, 8);
    }
    else if (featureType == "bluebins")
    {
        // Find the feature vector for the target image
        bool targetFound = false;
        for (size_t i = 0; i < filenames.size(); ++i)
        {
            // std::cout<<std::string(filenames[i]);
            if (std::string(filenames[i]) == targetImageFilename)
            {

                target_feature_vector = data[i];
                targetFound = true;
                break;
            }
        }
        if (!targetFound)
        {
            std::cerr << "Feature vector for target image not found." << std::endl;
            return -1;
        }
        target_features = computeBlueChromaticityHistogram(targetImgROI, 16);
    }
    else
    {
        printf("Invalid feature type: %s\n", featureType.c_str());
        exit(-1);
    }

    // this will store filenames and their distances in pairs;
    std::vector<std::pair<std::string, double>> distances;

    dirp = opendir(dirname);
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    int img_counter = 0;
    while ((dp = readdir(dirp)) != NULL)
    {
        img_counter += 1;
        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif"))
        {
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            // printf("full path name %s",buffer);

            cv::Mat image = cv::imread(buffer);
            double distance = -1.0;
            if (featureType == "baseline")
            {
                std::vector<float> feature_vector = computeBaselineFeatures(image);
                distance = sumSquaredDistance(target_feature_vector, feature_vector);
            }
            else if (featureType == "histogram")
            {
                cv::Mat features = computeRGChromaticityHistogram(image, 16);
                distance = histogramIntersection(target_features, features);
            }
            else if (featureType == "multihistogram")
            {
                std::pair<cv::Mat, cv::Mat> features_hist = computeSpatialHistograms(image, 8);
                distance = combinedHistogramDistance(target_hist, features_hist);
            }
            else if (featureType == "dnn")
            {
                std::vector<float> feature_vector;
                bool featureFound = false;
                for (size_t i = 0; i < filenames.size(); ++i)
                {
                    if (std::string(filenames[i]) == dp->d_name)
                    {
                        feature_vector = data[i];
                        featureFound = true;
                        break;
                    }
                }
                if (!featureFound)
                {
                    std::cerr << "Feature vector for target image not found." << std::endl;
                    return -1;
                }
                distance = cosineDistance(target_feature_vector, feature_vector);
            }
            else if (featureType == "texture")
            {
                std::pair<cv::Mat, cv::Mat> features_hist = computeSpatialHistograms_texture(image, 8);
                distance = combinedHistogramDistance_texture(target_hist, features_hist);
            }
            else if (featureType == "gabor")
            {
                std::pair<cv::Mat, cv::Mat> features_hist = computeSpatialHistograms_gabor(image, 8);
                distance = combinedHistogramDistance_texture(target_hist, features_hist);
            }
            else if (featureType == "grass")
            {
                cv::Mat features = computeGrassChromaticityHistogram(image, 16);
                double edgeDensity2 = computeEdgeDensity(image);
                double grassCoverage2 = computeGrassCoverage(image);
                // Find DNN feature vector
                std::vector<float> feature_vector;
                bool featureFound = false;
                for (size_t i = 0; i < filenames.size(); ++i)
                {
                    if (std::string(filenames[i]) == dp->d_name)
                    {
                        feature_vector = data[i];
                        featureFound = true;
                        break;
                    }
                }
                if (!featureFound)
                {
                    std::cerr << "Feature vector for feature image not found." << std::endl;
                    return -1;
                }
                distance = compositeDistance(target_features, features, edgeDensity, edgeDensity2, grassCoverage, grassCoverage2, target_feature_vector, feature_vector);
                cout << "Distance: " << distance << endl;
            }
            else if (featureType == "bluebins")
            {
                cv::Mat features = computeBlueChromaticityHistogram(image, 16);
                // Find DNN feature vector
                std::vector<float> feature_vector;
                bool featureFound = false;
                for (size_t i = 0; i < filenames.size(); ++i)
                {
                    if (std::string(filenames[i]) == dp->d_name)
                    {
                        feature_vector = data[i];
                        featureFound = true;
                        break;
                    }
                }
                if (!featureFound)
                {
                    std::cerr << "Feature vector for feature image not found." << std::endl;
                    return -1;
                }
                distance = computeDistanceBins(target_features, features, target_feature_vector, feature_vector, 0.5, 0.5);
            }
            else
            {
                printf("Invalid feature type: %s\n", featureType.c_str());
                exit(-1);
            }
            if (distance >= 0)
            { // Ensure distance is valid
                // cout << "Distance: " << distance << endl;
                distances.push_back(std::make_pair(buffer, distance));
            }
        }
    std:
        cout << "Image Count: " << img_counter << endl;
    }
    closedir(dirp);

    if (featureType == "baseline" ||
        featureType == "multihistogram" ||
        featureType == "texture" ||
        featureType == "dnn" ||
        featureType == "grass" ||
        featureType == "bluebins" ||
        featureType == "gabor")
    {
        sort(distances, true);
    }
    else if (featureType == "histogram")
    {
        sort(distances, false);
    }

    // Displaying top N matches, starting from the second match to avoid the target image itself if present
    for (int i = 1; i <= N && i < distances.size(); ++i)
    {
        cv::Mat picture = cv::imread(distances[i].first.c_str());
        cv::imshow("Picture", picture);
        cv::waitKey(0);
        std::cout << "Distance: " << distances[i].second << endl;
    }

    return 0;
}
#endif
