#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "chessboardcorner.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
     // Wait for a keystroke in the window
    cv::VideoCapture capdev(0);
    cv::Mat frame;
    cv::Size boardSize(9,6);
    cv::namedWindow("Display Window",WINDOW_NORMAL);
    
    if(!capdev.isOpened()){
        printf("Unable to open the video camera");
        return -1;
    }
    std::vector<cv::Point2f> corner_set;
    std::vector<cv::Vec3f> point_set;
    std::vector<cv::Vec3f> point_set1;
    std::vector<std::vector<cv::Vec3f> > point_list;
    std::vector<std::vector<cv::Point2f> > corner_list;

    //Initializing 3x3 Matrix

    cv::Mat camera_matrix=cv::Mat::eye(3,3, CV_64FC1);
    camera_matrix.at<double>(0,2)=frame.cols/2;
    camera_matrix.at<double>(1,2)=frame.rows/2;
    cv::Mat distortion_coefficients = cv::Mat::zeros(5, 1, CV_64FC1);
    int flag=0;
    while(true)
    {
        capdev >> frame;
        char k = waitKey(10);
        if(frame.empty()){
            printf("Blank Frame grabbed");
            return -1;
        }
        //Task 1
        bool foundCorners = drawchessboardcorner(frame,boardSize, corner_set);      //to find and display chessboard corners
        

        //Task 2
        
        if(k=='s' && !corner_set.empty())
        {
            std::cout<<"Number of the corners"<<corner_set.size()<<std::endl;
            std::cout<<"First Coordinate: "<<corner_set[0]<<std::endl;   
            corner_list.push_back(corner_set);

            //Saving list of corner sets
            for (size_t i = 0; i < corner_set.size(); ++i) 
            {
                int x, y;
                // std::cout<< "Horizontal orientation (9x6)"<<std::endl;
                x = static_cast<int>(i % boardSize.width);
                y = static_cast<int>(i / boardSize.width);
                point_set.push_back(cv::Vec3f(x, -y, 0.0f));
                // std::cout<<"Calibration Points: "<<point_set[i]<<std::endl;
            }
            // point_list.push_back(point_set);
            // //std::cout<<"Calibration Points: "<<point_list<<std::endl;
            // point_set.clear();
            // //std::cout << "Calibration Points:" << std::endl;
            
            if (!point_set.empty()) {
                point_list.push_back(point_set);

                // Clear point_set for the next iteration
                point_set.clear();

                // Display the size of point_list
                std::cout << "Size of point_list: " << point_list.size() << " (Number of frames)" << std::endl;

                // Display the size of each frame in point_list
                for (size_t i = 0; i < point_list.size(); ++i) {
                    std::cout << "Frame " << i << " size: " << point_list[i].size() << " (Number of points)" << std::endl;
                }
            } 
            else 
            {
                std::cout << "Point_set is empty. Skipping..." << std::endl;
            }
            
            std::cout<<"Callibration points saved"<<std::endl;
            flag+=1;

        }
        // Task 3
        if(flag>=5)
        {
            calibrateCameraAndSaveParameters(point_list, corner_list, frame.size(), camera_matrix, distortion_coefficients);            

            //Task 4
            if (foundCorners) {
                calculatePose(corner_set, camera_matrix, distortion_coefficients, boardSize);
            }
        }
        
        cv::imshow("Display Window",frame);
        if(k=='q')
        {                 //If the user presses 'q', end the program
            break;
        }
    }
    capdev.release();
    cv::destroyAllWindows();
    return 0;
}