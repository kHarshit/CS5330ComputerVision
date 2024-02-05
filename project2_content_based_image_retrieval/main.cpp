/**
 * author: Harshit Kumar, Khushi Neema
 * date: Jan 29, 2024
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include "include/imgProcess.h"
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
    target_image=cv::imread("/home/kharshit/Khushi/olympus/pic.1016.jpg");
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