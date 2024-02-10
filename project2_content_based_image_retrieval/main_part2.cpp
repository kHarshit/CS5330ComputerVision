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
#include "include/distance.h"
#include "include/feature.h"
#include "include/csv_util.h"

using namespace std;

// If you want to write data to csv, set it to 1
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

/**
 * @brief Select a region of interest (ROI) from an image
 * 
 * @param image Input image
 * @return cv::Rect ROI
*/
cv::Rect selectROI(const cv::Mat &image)
{
    // Let the user select a region of interest on the image
    cv::Rect roi = cv::selectROI("Select ROI", image);
    cv::destroyWindow("Select ROI");
    return roi;
}

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
                append_image_data_csv(argv[2], const_cast<char *>(dp->d_name), features, reset_file);
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
        double distance = sumSquaredDistance(target_feature_vector, data[i]);
        // double distance = cosineDistance(target_feature_vector, data[i]);
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