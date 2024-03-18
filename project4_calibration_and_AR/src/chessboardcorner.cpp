/**
 * author: Harshit Kumar, Khushi Neema
 * date: Mar 5th, 2024
 * purpose: Implements various Calibration and Augmented Reality functionalities
 *
 */
#include "chessboardcorner.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;

bool drawchessboardcorner(cv::Mat frame, cv::Size boardSize, std::vector<cv::Point2f> &corner_set)
{
    cv::Mat gray;
    cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);

    bool found = findChessboardCorners( gray, boardSize, corner_set, cv::CALIB_CB_ADAPTIVE_THRESH );
    if(found)
    {    
        cv::cornerSubPix(gray,corner_set,cv::Size(11,11),cv::Size(-1,-1),cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
        //std::cout<<corner_set;
        cv::drawChessboardCorners(frame,boardSize,corner_set,true);
        
    }
    return found;
}

void saveCalibrationPoints(vector<cv::Point2f>& corner_set, vector<vector<cv::Point2f>>& corner_list, vector<cv::Vec3f>& point_set, vector<vector<cv::Vec3f>>& point_list, int& flag, Size boardSize) {
    if(!corner_set.empty())
    {
        std::cout<<"Number of the corners"<<corner_set.size()<<std::endl;
        std::cout<<"First Coordinate: "<<corner_set[0]<<std::endl;   
        corner_list.push_back(corner_set);

        for (size_t i = 0; i < corner_set.size(); ++i) 
        {
            int x, y;
            x = static_cast<int>(i % boardSize.width);
            y = static_cast<int>(i / boardSize.width);
            point_set.push_back(cv::Vec3f(x, -y, 0.0f));
        }

        if (!point_set.empty()) {
            point_list.push_back(point_set);
            point_set.clear();
            std::cout << "Size of point_list: " << point_list.size() << " (Number of frames)" << std::endl;

            for (size_t i = 0; i < point_list.size(); ++i) {
                std::cout << "Frame " << i << " size: " << point_list[i].size() << " (Number of points)" << std::endl;
            }
        } 
        else 
        {
            std::cout << "Point_set is empty. Skipping..." << std::endl;
        }

        std::cout<<"Calibration points saved"<<std::endl;
        flag+=1;
    }
}

void calibrateCameraAndSaveParameters(std::vector<std::vector<cv::Vec3f>>& point_list, std::vector<std::vector<cv::Point2f>>& corner_list, cv::Size frame_size, cv::Mat& camera_matrix, cv::Mat& distortion_coefficients) {
    // Calibrating camera
    std::cout << "Previous Calibrated Camera" << camera_matrix;
    std::cout << "Previous Distortion Coefficients" << distortion_coefficients;
    std::vector<cv::Mat> rvecs, tvecs;
    double error = cv::calibrateCamera(point_list, corner_list, frame_size, camera_matrix, distortion_coefficients, rvecs,tvecs, cv::CALIB_FIX_ASPECT_RATIO);

    std::cout << "Camera Matrix (after calibration):\n" << camera_matrix << std::endl;
    std::cout << "Distortion Coefficients (after calibration):\n" << distortion_coefficients << std::endl;
    std::cout << "Reprojection Error: " << error << std::endl;

    // Write intrinsic parameters to a file
    cv::FileStorage fs("intrinsic_parameters.yml", cv::FileStorage::WRITE);
    fs << "camera_matrix" << camera_matrix;
    fs << "distortion_coefficients" << distortion_coefficients;
    fs.release();
}

void calculatePose(const std::vector<cv::Point2f>& corner_set, const cv::Mat& camera_matrix, const cv::Mat& distortion_coefficients, const cv::Size& boardSize, cv::Mat& rvec, cv::Mat& tvec) {
    // Define object points in real world space
    std::vector<cv::Point3f> object_points;
    for(int i = 0; i < boardSize.height; ++i)
        for(int j = 0; j < boardSize.width; ++j)
            object_points.push_back(cv::Point3f(j, i, 0.0f));

    // Get board's pose
    cv::solvePnP(object_points, corner_set, camera_matrix, distortion_coefficients, rvec, tvec);
}

void projectPointsAndDraw(const std::vector<cv::Point2f>& corner_set, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& camera_matrix, const cv::Mat& distortion_coefficients, const cv::Size& boardSize, cv::Mat& image) {
    // Define object points in real world space
    // std::vector<cv::Point3f> object_points;
    // for(int i = 0; i < boardSize.height; ++i)
    //     for(int j = 0; j < boardSize.width; ++j)
    //         object_points.push_back(cv::Point3f(j, i, 0.0f));

    // // Project 3D points to image plane
    // std::vector<cv::Point2f> projected_points;
    // cv::projectPoints(object_points, rvec, tvec, camera_matrix, distortion_coefficients, projected_points);

    // Draw projected points on the image
    // for (size_t i = 0; i < projected_points.size(); ++i) {
    //     cv::circle(image, projected_points[i], 3, cv::Scalar(0, 0, 255), -1);
    // }

    // Project 3D Points to 2D or Draw 3D Axes
    std::vector<cv::Point3f> axisPoints = {
        cv::Point3f(0.0f, 0.0f, 0.0f),  // Origin
        cv::Point3f(2.0f, 0.0f, 0.0f),  // X Axis
        cv::Point3f(0.0f, 2.0f, 0.0f),  // Y Axis
        cv::Point3f(0.0f, 0.0f, -2.0f)  // Z Axis (into the board)
    };
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(axisPoints, rvec, tvec, camera_matrix, distortion_coefficients, imagePoints);

    // Draw the axes lines
    cv::line(image, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 3);  // X Axis in Red
    cv::line(image, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 3);  // Y Axis in Green
    cv::line(image, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 3);  // Z Axis in Blue
}

void createObject(const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& camera_matrix, const cv::Mat& distortion_coefficients, const cv::Size& boardSize, cv::Mat& image)
{

    std::vector<cv::Point3f> person_standing ={
        cv::Point3f(4.0f,3.0f,0.0f),    //legs left
        cv::Point3f(5.0f,3.0f,0.0f),   //legs right
        cv::Point3f(4.0f,3.0f,-2.0f),    //body left start
        cv::Point3f(5.0f,3.0f,-2.0f),    //body right start
        cv::Point3f(4.0f,3.0f,-6.0f),   //body left end
        cv::Point3f(5.0f,3.0f,-6.0f),   // body right end
        cv::Point3f(4.5f,3.0f,-6.0f),   // neck start
        cv::Point3f(4.5f,3.0f,-6.25f),  // neck end
        cv::Point3f(4.5f,3.0f,-7.25f),  // head center
        cv::Point3f(5.75f,2.50f,-5.0f), //elbow right
        cv::Point3f(7.0f,2.80f,-6.25f),  //wrist right
        cv::Point3f(2.75f,2.40f,-4.75f),    //elbow left
        cv::Point3f(3.75f,3.7f,-3.8f)   //wrist left
    };

        std::vector<cv::Point2f> projected_points;
        cv::projectPoints(person_standing, rvec, tvec, camera_matrix, distortion_coefficients, projected_points);


        // Draw the person
        cv::line(image, projected_points[0], projected_points[2], cv::Scalar(0, 255, 0), 2);    // Head to Body
        cv::line(image, projected_points[1], projected_points[3], cv::Scalar(0, 255, 0), 2);    // Body to Left shoulder
        cv::line(image, projected_points[2], projected_points[3], cv::Scalar(0, 255, 0), 2);    // Body to Right shoulder
        cv::line(image, projected_points[2], projected_points[4], cv::Scalar(0, 255, 0), 2);    // Left shoulder to Left hip
        cv::line(image, projected_points[3], projected_points[5], cv::Scalar(0, 255, 0), 2);    // Right shoulder to Right hip
        cv::line(image, projected_points[4], projected_points[5], cv::Scalar(0, 255, 0), 2);    // Left shoulder to Left hand
        cv::line(image, projected_points[6], projected_points[7], cv::Scalar(0, 255, 0), 2);    // Right shoulder to Right hand
        // cv::line(image, projected_points[8], projected_points[9], cv::Scalar(0, 255, 0), 2);    // Left hip to Left foot
        // cv::line(image, projected_points[8], projected_points[10], cv::Scalar(0, 255, 0), 2);
        // cv::line(image, projected_points[9], projected_points[11], cv::Scalar(0, 255, 0), 2);
        cv::line(image, projected_points[5], projected_points[9], cv::Scalar(0, 255, 0), 2);
        cv::line(image, projected_points[9], projected_points[10], cv::Scalar(0, 255, 0), 2);
        cv::line(image, projected_points[4], projected_points[11], cv::Scalar(0, 255, 0), 2);
        cv::line(image, projected_points[11], projected_points[12], cv::Scalar(0, 255, 0), 2);
        cv::circle(image,projected_points[8],40.0,cv::Scalar(0, 255, 0), 2);

   
    std::vector<cv::Point3f> house_coords = {
        cv::Point3f(0.0f, 0.0f, 0.0f),    // Base left
        cv::Point3f(6.0f, 0.0f, 0.0f),    // Base right
        cv::Point3f(6.0f, 0.0f, -4.0f),   // Base right back
        cv::Point3f(0.0f, 0.0f, -4.0f),   // Base left back
        cv::Point3f(3.0f, 0.5f, -6.0f),   // Roof center back
        cv::Point3f(3.0f, 4.0f, -2.0f),   // Roof top
        cv::Point3f(2.0f, 0.0f, 0.0f),   // Door left
        cv::Point3f(4.0f, 0.0f, 0.0f),   // Door right
        cv::Point3f(2.0f, 0.0f, -1.5f),   // Door top left
        cv::Point3f(4.0f,0.0f,-1.5f),   //Door top right
        
        cv::Point3f(3.0f,0.5f,-5.0f),

        cv::Point3f(0.0f, 1.0f, 0.0f),    // Base left ahead
        cv::Point3f(6.0f, 1.0f, 0.0f),      //Base right ahead
        cv::Point3f(6.0f, 1.0f, -4.0f),   // Base right back
        cv::Point3f(0.0f, 1.0f, -4.0f), 

        cv::Point3f(2.0f, 1.0f, 0.0f),   // Door left
        cv::Point3f(4.0f, 1.0f, 0.0f),   // Door right
        cv::Point3f(4.0f,1.0f,-1.5f),   // Door top left
        cv::Point3f(2.0f, 1.0f, -1.5f)
        
    };
    std::vector<cv::Point2f> projected_points1;
    cv::projectPoints(house_coords, rvec, tvec, camera_matrix, distortion_coefficients, projected_points1);

    // Draw the base
    cv::line(image, projected_points1[0], projected_points1[1], cv::Scalar(255, 0, 0), 2); // Base left to right
    cv::line(image, projected_points1[1], projected_points1[2], cv::Scalar(255, 0, 0), 2); // Base right to right back
    cv::line(image, projected_points1[2], projected_points1[3], cv::Scalar(255, 0, 0), 2); // Base right back to left back
    cv::line(image, projected_points1[3], projected_points1[0], cv::Scalar(255, 0, 0), 2); // Base left back to left

    //Draw base 2
    cv::line(image, projected_points1[11], projected_points1[12], cv::Scalar(255, 0, 0), 2); // Base left to right
    cv::line(image, projected_points1[12], projected_points1[13], cv::Scalar(255, 0, 0), 2); // Base right to right back
    cv::line(image, projected_points1[13], projected_points1[14], cv::Scalar(255, 0, 0), 2); // Base right back to left back
    cv::line(image, projected_points1[14], projected_points1[11], cv::Scalar(255, 0, 0), 2); // Base left back to left

    //Connecting both bases
    cv::line(image, projected_points1[0], projected_points1[11], cv::Scalar(255, 0, 0), 2); 
    cv::line(image, projected_points1[1], projected_points1[12], cv::Scalar(255, 0, 0), 2);
    cv::line(image, projected_points1[2], projected_points1[13], cv::Scalar(255, 0, 0), 2);  
    cv::line(image, projected_points1[3], projected_points1[14], cv::Scalar(255, 0, 0), 2); 
    // Draw the roof
    //cv::line(image, projected_points1[4], projected_points1[5], cv::Scalar(255, 0, 0), 2); // Roof center back to top
    cv::line(image, projected_points1[3], projected_points1[4], cv::Scalar(255, 0, 0), 2); // Base left back to roof center back
    cv::line(image, projected_points1[2], projected_points1[4], cv::Scalar(255, 0, 0), 2); // Base right back to roof center back
    cv::line(image, projected_points1[13], projected_points1[4], cv::Scalar(255, 0, 0), 2); 
    cv::line(image, projected_points1[14], projected_points1[4], cv::Scalar(255, 0, 0), 2); 

    // Draw the door
    cv::line(image, projected_points1[6], projected_points1[7], cv::Scalar(255, 0, 0), 2); // Door left to right
    cv::line(image, projected_points1[6], projected_points1[8], cv::Scalar(255, 0, 0), 2); // Door left to top
    cv::line(image, projected_points1[7], projected_points1[9], cv::Scalar(255, 0, 0), 2); // Door right to top
    cv::line(image, projected_points1[8], projected_points1[9], cv::Scalar(255, 0, 0), 2); //Door top connect

    //Draw the door 2
     cv::line(image, projected_points1[15], projected_points1[16], cv::Scalar(255, 0, 0), 2); // Door left to right
    cv::line(image, projected_points1[16], projected_points1[17], cv::Scalar(255, 0, 0), 2); // Door left to top
    cv::line(image, projected_points1[17], projected_points1[18], cv::Scalar(255, 0, 0), 2); // Door right to top
    cv::line(image, projected_points1[18], projected_points1[15], cv::Scalar(255, 0, 0), 2); //Door top connect

    //Connect doors
    //Connecting both bases
    cv::line(image, projected_points1[6], projected_points1[15], cv::Scalar(255, 0, 0), 2); 
    cv::line(image, projected_points1[7], projected_points1[16], cv::Scalar(255, 0, 0), 2);
    cv::line(image, projected_points1[8], projected_points1[18], cv::Scalar(255, 0, 0), 2);  
    cv::line(image, projected_points1[9], projected_points1[17], cv::Scalar(255, 0, 0), 2); 


    cv::circle(image,projected_points1[10],20.0,cv::Scalar(255, 0, 0), 2);
    //cv::circle(image,projected_points1[15],20.0,cv::Scalar(255, 0, 0), 2);
}

