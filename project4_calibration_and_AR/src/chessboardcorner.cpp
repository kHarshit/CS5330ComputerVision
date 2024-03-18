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

void calibrateCameraAndSaveParameters(std::vector<std::vector<cv::Vec3f>>& point_list, std::vector<std::vector<cv::Point2f>>& corner_list, cv::Size frame_size, cv::Mat& camera_matrix, cv::Mat& distortion_coefficients) {
    // Calibrating camera
    std::cout << "Previous Calibrated Camera" << camera_matrix;
    std::cout << "Previous Distortion Coefficients" << distortion_coefficients;

    double error = cv::calibrateCamera(point_list, corner_list, frame_size, camera_matrix, distortion_coefficients, cv::noArray(), cv::noArray(), cv::CALIB_FIX_ASPECT_RATIO);

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
    //Creating a BUILDING block object 

    // std::vector<cv::Point3f> house_points = {
    // // Define the vertices of the house relative to its center
    // cv::Point3f(-1.5f, -1.5f, 0.0f),     // Bottom left corner of the house
    // cv::Point3f(1.5f, -1.5f, 0.0f),      // Bottom right corner of the house
    // cv::Point3f(1.5f, 1.5f, 0.0f),       // Top right corner of the house
    // cv::Point3f(-1.5f, 1.5f, 0.0f),      // Top left corner of the house
    // cv::Point3f(0.0f, 3.0f, 0.0f)        // Roof peak of the house
    // };

    // std::vector<cv::Point2f> projected_points;
    // cv::projectPoints(house_points, rvec, tvec, camera_matrix, distortion_coefficients, projected_points);

    // cv::line(image, projected_points[0], projected_points[1], cv::Scalar(0, 0, 255), 2);  // Bottom
    // cv::line(image, projected_points[1], projected_points[2], cv::Scalar(0, 0, 255), 2);  // Right
    // cv::line(image, projected_points[2], projected_points[3], cv::Scalar(0, 0, 255), 2);  // Top
    // cv::line(image, projected_points[3], projected_points[0], cv::Scalar(0, 0, 255), 2);  // Left
    // cv::line(image, projected_points[0], projected_points[4], cv::Scalar(0, 0, 255), 2);  // Bottom-left to Roof
    // cv::line(image, projected_points[1], projected_points[4], cv::Scalar(0, 0, 255), 2);  // Bottom-right to Roof
    // cv::line(image, projected_points[2], projected_points[4], cv::Scalar(0, 0, 255), 2);  // Top-right to Roof
    // cv::line(image, projected_points[3], projected_points[4], cv::Scalar(0, 0, 255), 2);  // Top-left to Roof

    // Define the center of the chessboard
    cv::Point3f chessboard_center(4.0f, 3.0f, 0.0f);
    // Scale factor
    float scale_factor = 6.0f;

    // Define the 3D coordinates for a person standing on the chessboard
    std::vector<cv::Point3f> person_points = {
        // Head
    cv::Point3f(chessboard_center.x, chessboard_center.y, scale_factor * 1.8f),        // Head center
    
    // Body
    cv::Point3f(chessboard_center.x, chessboard_center.y, scale_factor * 1.2f),        // Body center
    cv::Point3f(chessboard_center.x - 0.3f * scale_factor, chessboard_center.y, scale_factor * 0.8f), // Left shoulder
    cv::Point3f(chessboard_center.x + 0.3f * scale_factor, chessboard_center.y, scale_factor * 0.8f), // Right shoulder
    cv::Point3f(chessboard_center.x - 0.3f * scale_factor, chessboard_center.y, scale_factor * 0.4f), // Left hip
    cv::Point3f(chessboard_center.x + 0.3f * scale_factor, chessboard_center.y, scale_factor * 0.4f), // Right hip
    
    // Arms
    cv::Point3f(chessboard_center.x - 0.6f * scale_factor, chessboard_center.y, scale_factor * 1.0f), // Left hand
    cv::Point3f(chessboard_center.x + 0.6f * scale_factor, chessboard_center.y, scale_factor * 1.0f), // Right hand
    
    // Legs
    cv::Point3f(chessboard_center.x - 0.3f * scale_factor, chessboard_center.y, 0.0f), // Left foot
    cv::Point3f(chessboard_center.x + 0.3f * scale_factor, chessboard_center.y, 0.0f)  // Right foot
    };


    std::vector<cv::Point3f> person_standing ={
        cv::Point3f(4.0f,3.0f,0.0f),    //legs left
        cv::Point3f(5.0f,3.0f,0.0f),   //legs right
        cv::Point3f(4.0f,3.0f,-2.0f),    //body left start
        cv::Point3f(5.0f,3.0f,-2.0f),    //body right start
        cv::Point3f(4.0f,3.0f,-6.0f),   //body left end
        cv::Point3f(5.0f,3.0f,-6.0f),   // body right end
        cv::Point3f(4.5f,3.0f,-6.0f),
        cv::Point3f(4.5f,3.0f,-6.25f),
        cv::Point3f(4.5f,3.0f,-7.25f),
        cv::Point3f(4.25f,3.0f,-6.25f),
        cv::Point3f(4.75f,3.0f,-6.25f),
        cv::Point3f(4.25f,3.0f,-7.25f),
        cv::Point3f(4.75f,3.0f,-7.25f),
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
        // cv::line(image, projected_points[11], projected_points[10], cv::Scalar(0, 255, 0), 2);
        cv::circle(image,projected_points[8],40.0,cv::Scalar(0, 255, 0), 2);
            // Right hip to Right foot


}