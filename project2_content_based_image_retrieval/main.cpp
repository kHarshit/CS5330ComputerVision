/**
 * author: Harshit Kumar, Khushi Neema
 * date: Jan 29, 2024
 */

#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
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

void Sort(std::vector<std::pair<std::string, double> >& matches) {
    int n = matches.size();
    for (int i = 0; i < n - 1; ++i) {
        int minIndex = i;
        for (int j = i + 1; j < n; ++j) {
            if (matches[j].second < matches[minIndex].second) {
                minIndex = j;
            }
        }
        if (minIndex != i) {
            std::swap(matches[i], matches[minIndex]);
        }
    }
}

#if !RUN_PART1

int main(int argc,char *argv[])
{

    #if 0
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <directory path> <output CSV file>" << std::endl;
        return -1;
    }

    DIR *dirp;
    struct dirent *dp;
    dirp = opendir(argv[1]);
    if (dirp == NULL) {
        std::cout << "Cannot open directory " << argv[1] << std::endl;
        return -1;
    }

    bool reset_file = true; // Reset the file for the first image
    while ((dp = readdir(dirp)) != NULL) {
        if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif")) {
            std::string filepath = std::string(argv[1]) + "/" + std::string(dp->d_name);
            cv::Mat image = cv::imread(filepath);
            if (!image.empty()) {
                cv::Mat features = computeBaselineFeatures(image);
                // Convert features to a vector<float>
                std::vector<float> feature_vector(features.begin<float>(), features.end<float>());
                append_image_data_csv(argv[2], const_cast<char *>(filepath.c_str()), feature_vector, reset_file);
                reset_file = false; // Only reset the file once, for the first image
            }
        }
    }

    closedir(dirp);
    #endif

    #if 1

            if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <target image> <feature vector file> <N>" << std::endl;
        return -1;
    }

    cv::Mat target_image = cv::imread(argv[1]);
    if (target_image.empty()) {
        std::cout << "Could not read the target image." << std::endl;
        return -1;
    }

    int N = std::atoi(argv[3]);

    // Assuming computeBaselineFeatures returns a cv::Mat that needs to be converted to a std::vector<float>
    cv::Mat target_features_mat = computeBaselineFeatures(target_image);
    std::vector<float> target_feature_vector(target_features_mat.begin<float>(), target_features_mat.end<float>());

    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;
    if (read_image_data_csv(argv[2], filenames, data, 0) != 0) {
        std::cerr << "Error reading feature vector file." << std::endl;
        return -1;
    }

    std::vector<std::pair<std::string, float>> distances;
    for (size_t i = 0; i < data.size(); ++i) {
        float distance = computeDistance(target_feature_vector, data[i]);
        if (distance >= 0) { // Ensure distance is valid
            cout << "Distance: " << distance << endl;
            distances.push_back(std::make_pair(std::string(filenames[i]), distance));
        }
    }

    // Sorting the distances in ascending order
    std::sort(distances.begin(), distances.end(), [](const std::pair<std::string, float> &a, const std::pair<std::string, float> &b) {
        return a.second < b.second;
    });

    // Displaying top N matches, starting from the second match to avoid the target image itself if present
    for (int i = 1; i <= N && i < distances.size(); ++i) {
        std::cout << "Distance: " << distances[i].second << ", File: " << distances[i].first << std::endl;
    }


    #endif

    #if 0
    char dirname[256];
    char buffer[256];
    int flag=0;
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;
    std::vector<std::pair<std::string, double> > matches; 
    char filename[]="Feature_Vector.csv";
    cv::Mat target_image,target_features;
    std::vector<float> featureVector;
    if (argc < 2)
    {
        printf("usage: %s <directory path>\n", argv[0]);
        exit(-1);
    }
    strcpy(dirname, argv[1]);
    printf("Processing directory %s\n", dirname);


    //Reading the target image and computing its features
    target_image=cv::imread("/home/kharshit/Khushi/olympus/pic.0503.jpg");
    //cv::imshow("Target Image",target_image);
    //cv::waitKey(0);
    cout << "Computing target image features\n";
    target_features=computeBaselineFeatures(target_image);

    dirp = opendir(dirname);
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    #if WRITE_CSV
    cout << "Writing to CSV file\n";
    while ((dp = readdir(dirp)) != NULL)
    {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif"))
        {
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            //printf("full path name %s",buffer);
        
        cv::Mat current_image=cv::imread(buffer);
        cv::Mat current_features=computeBaselineFeatures(current_image);
        
        //Convert cv::Mat to std::vector<float>
        
        featureVector.assign(current_features.begin<float>(), current_features.end<float>());
         if(flag==0){
             flag=1;
            int status=append_image_data_csv(filename,dp->d_name,featureVector,1);
        }
        std::vector<float> featureVector;
        current_features.reshape(1, 1).convertTo(featureVector, CV_32F);
        int status=append_image_data_csv(filename,dp->d_name,featureVector,0);
        if(status!=0){
            printf("Error in writing information in file");
        }
        }
             
    }
    #endif
    cout << "Reading from CSV file\n";
    std::vector<char*> filenames;
    std::vector<std::vector<float> >  data;
    if (read_image_data_csv(filename, filenames, data, 0) != 0) {
        printf("Error: Unable to read image data from CSV file.\n");
        return -1;
    }
//     for (int i = 0; i < data.size(); ++i) {
//     std::cout << "Filename: " << filenames[i] << std::endl;
//     std::cout << "Feature Vector:";
//     cv::Mat featureMat(256, 320, CV_32F);
//     memcpy(featureMat.data, data[i].data(), data[i].size() * sizeof(float));
//     //featureMat=featureMat.reshape(1,1);
//     std::cout << featureMat.rows <<" ";
//     std::cout<<featureMat.cols;
//     std::cout << std::endl;
//     break;
// }
    
    
    cout << "Computing distances\n";
    //Compute distances between target image features and features of all other images
    for (int i = 0; i < filenames.size(); ++i) {
        cout << filenames[i] << endl;
        
        if (strcmp(filenames[i], "pic.0503.jpg") != 0) {  // Exclude the target image itself
            // cv::Mat featureMat(target_features.rows,target_features.cols, CV_32F);
            // memcpy(featureMat.data, data[i].data(), data[i].size() * sizeof(float));
            // std::cout << " Target Size " << target_features.rows << " "<<target_features.cols;
            //std::cout << " Input Size " << featureMat.rows << " "<<featureMat.cols;
            //std::cout << "Type of Input " << featureMat.type();
            std::cout<< "Type of Target " << target_features.type();
            
            double distance = computeDistance(target_features, cv::Mat(data[i]).reshape(256,320));
            matches.push_back(std::make_pair(std::string(filenames[i]), distance));
        }
    }

    // Sort matches based on distance
    Sort(matches);

    // Print top N matches
    int N = 5;  // Number of closest matches to print
    std::cout << "Top " << N << " closest matches:" << std::endl;
    for (int i = 0; i < N && i < matches.size(); ++i) {
        std::cout << "Filename: " << matches[i].first << ", Distance: " << matches[i].second << std::endl;
    }

    // Clean up memory
    for (int i = 0; i < filenames.size(); ++i) {
        delete[] filenames[i];
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
    // this will store filenames and their distances in pairs;
    std::vector<std::pair<std::string, double> > matches; 

    cv::Mat target_image,target_features;
    if (argc < 2)
    {
        printf("usage: %s <directory path>\n", argv[0]);
        exit(-1);
    }
    
    strcpy(dirname, argv[1]);
    printf("Processing directory %s\n", dirname);

    //Reading the target image and computing its features
    target_image=cv::imread("/Users/harshit/Downloads/olympus/pic.1016.jpg");
    cv::imshow("Target Image",target_image);
    cv::waitKey(0);

    printf("Computing its features : ");
    target_features=computeBaselineFeatures(target_image);
    // Convert features to a vector<float>
    std::vector<float> target_feature_vector(target_features.begin<float>(), target_features.end<float>());
    // store distances
    std::vector<std::pair<std::string, float>> distances;

    dirp = opendir(dirname);
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    while ((dp = readdir(dirp)) != NULL)
    {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif"))
        {
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            //printf("full path name %s",buffer);
        
        cv::Mat image=cv::imread(buffer);
        cv::Mat features=computeBaselineFeatures(image);
        std::vector<float> feature_vector(features.begin<float>(), features.end<float>());
        double distance=computeDistance(feature_vector,target_feature_vector);

        cout << "Distance: " << distance << endl;
        distances.push_back(std::make_pair(buffer, distance));
        matches.push_back(std::make_pair(buffer,distance));
        }
    }
    closedir(dirp);

    // Sorting the distances in ascending order
    std::sort(distances.begin(), distances.end(), [](const std::pair<std::string, float> &a, const std::pair<std::string, float> &b) {
        return a.second < b.second;
    });

    int N = 3; // Number of top matches to display
    printf("Top %d matches:\n", N);
    // Displaying top N matches, starting from the second match to avoid the target image itself if present
    for (int i = 1; i <= N && i < distances.size(); ++i) {
        cv::Mat picture=cv::imread(matches[i].first.c_str());
        cv::imshow("Picture",picture);
        cv::waitKey(0);
        std::cout << "Distance: " << distances[i].second << endl;
    }

    return 0;
}
#endif

