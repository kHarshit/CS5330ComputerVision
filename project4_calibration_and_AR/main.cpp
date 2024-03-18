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

        //Task 2: Select calibration images
        if(k=='s' && !corner_set.empty())
        {
            saveCalibrationPoints(corner_set, corner_list, point_set, point_list, flag, boardSize);
        }
        #if 1
            flag=6;
            cv::FileStorage fs("intrinsic_parameters.yml", cv::FileStorage::READ);
            if (!fs.isOpened()) {
                std::cerr << "Error: Unable to open the file for reading." << std::endl;
                return 0;
            }
            fs["camera_matrix"] >> camera_matrix;
            fs["distortion_coefficients"] >> distortion_coefficients;
            fs.release();
        #endif
        // Task 3
        if(flag>=5)
        {
            // Task 4
            // calibrateCameraAndSaveParameters(point_list, corner_list, frame.size(), camera_matrix, distortion_coefficients);            

            //Task 5
            //foundCorners=false;
            if (foundCorners) {
                cv::Mat rvec, tvec;
                calculatePose(corner_set, camera_matrix, distortion_coefficients, boardSize, rvec, tvec);
                std::cout << "Rotation: " << rvec << std::endl;
                std::cout << "Translation: " << tvec << std::endl;

                projectPointsAndDraw(corner_set, rvec, tvec, camera_matrix, distortion_coefficients, boardSize, frame);

                // Blur the outside chessboard region
                blurOutsideChessboardRegion(boardSize, rvec, tvec, camera_matrix, distortion_coefficients, frame);
                // Blend the chessboard with grass
                cv::Mat texture = cv::imread("../grass.jpg");
                blendChessboardRegion(boardSize, rvec, tvec, camera_matrix, distortion_coefficients, frame, texture);
                createObject(rvec, tvec, camera_matrix, distortion_coefficients, boardSize, frame);
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