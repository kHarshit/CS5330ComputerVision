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
    std::vector<cv::Point3f> object_points;
    for(int i = 0; i < boardSize.height; ++i)
        for(int j = 0; j < boardSize.width; ++j)
            object_points.push_back(cv::Point3f(j, i, 0.0f));

    // Project 3D points to image plane
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, distortion_coefficients, projected_points);

    // Draw projected points on the image
    for (size_t i = 0; i < projected_points.size(); ++i) {
        cv::circle(image, projected_points[i], 3, cv::Scalar(0, 0, 255), -1);
    }
}