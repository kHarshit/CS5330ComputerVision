/**
 * author: Harshit Kumar, Khushi Neema
 * date: Jan 29, 2024
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include "include/imgProcess.h"
#include "include/csv_util.h"

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

int main(int argc,char *argv[])
{
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
    printf("Computing its features : ");
    target_features=computeBaselineFeatures(target_image);

    dirp = opendir(dirname);
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    // while ((dp = readdir(dirp)) != NULL)
    // {

    //     // check if the file is an image
    //     if (strstr(dp->d_name, ".jpg") ||
    //         strstr(dp->d_name, ".png") ||
    //         strstr(dp->d_name, ".ppm") ||
    //         strstr(dp->d_name, ".tif"))
    //     {
    //         strcpy(buffer, dirname);
    //         strcat(buffer, "/");
    //         strcat(buffer, dp->d_name);
    //         //printf("full path name %s",buffer);
        
    //     cv::Mat current_image=cv::imread(buffer);
    //     cv::Mat current_features=computeBaselineFeatures(current_image);
        
    // Convert cv::Mat to std::vector<float>
        
    //     featureVector.assign(current_features.begin<float>(), current_features.end<float>());
    //      if(flag==0){
    //          flag=1;
    //         int status=append_image_data_csv(filename,dp->d_name,featureVector,1);
    //     }
    //     std::vector<float> featureVector;
    //     current_features.reshape(1, 1).convertTo(featureVector, CV_32F);
    //     int status=append_image_data_csv(filename,dp->d_name,featureVector,0);
    //     if(status!=0){
    //         printf("Error in writing information in file");
    //     }
    //     }
             
    //}
    printf("Succesfully read all images");
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
    
    
    //Compute distances between target image features and features of all other images
    for (int i = 0; i < filenames.size(); ++i) {
        
        if (strcmp(filenames[i], "pic.0503.jpg") != 0) {  // Exclude the target image itself
            // cv::Mat featureMat(target_features.rows,target_features.cols, CV_32F);
            // memcpy(featureMat.data, data[i].data(), data[i].size() * sizeof(float));
            // std::cout << " Target Size " << target_features.rows << " "<<target_features.cols;
            //std::cout << " Input Size " << featureMat.rows << " "<<featureMat.cols;
            //std::cout << "Type of Input" << featureMat.type();
            std::cout<< "Type of Target" << target_features.type();
            
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

    return 0;
}

#if 0
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
    target_image=cv::imread("/home/kharshit/Khushi/olympus/pic.0503.jpg");
    cv::imshow("Target Image",target_image);
    cv::waitKey(0);
    printf("Computing its features : ");
    target_features=computeBaselineFeatures(target_image);

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
        double distance=computeDistance(features,target_features);

        matches.push_back(std::make_pair(buffer,distance));
        }
    }
    closedir(dirp);
    //Sorting the list in ascending order of distance : smallest distance : better match

    Sort(matches);

    int N = 3; // Number of top matches to display
    printf("Top %d matches:\n", N);

    //Not including the first matched image as that image will be equal to the target image;
    
    for (int i = 1; i < std::min(N+1, static_cast<int>(matches.size())); ++i) {
        printf("Distance: %.2f, File: %s\n", matches[i].second, matches[i].first.c_str());
        cv::Mat picture=cv::imread(matches[i].first.c_str());

        cv::imshow("Picture",picture);
        cv::waitKey(0);

    }

    return 0;
}
#endif

