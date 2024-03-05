/**
 * author: Harshit Kumar, Khushi Neema
 * date: Mar 5th, 2024
 * purpose: Implements various object detection algorithms
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

std::vector<cv::Point2f> Drawchessboardcorner(cv::Mat frame, cv::Size boardSize)
{
    std::vector<cv::Point2f> corner_set;
    cv::Mat gray;
    cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);

    bool found = findChessboardCorners( gray, boardSize, corner_set, cv::CALIB_CB_ADAPTIVE_THRESH );
    if(found)
    {    
        cv::cornerSubPix(gray,corner_set,cv::Size(11,11),cv::Size(-1,-1),cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
        //std::cout<<corner_set;
        cv::drawChessboardCorners(frame,boardSize,corner_set,true);
        std::cout<<"Number of the corners"<<corner_set.size()<<std::endl;
        std::cout<<"First Coordinate: "<<corner_set[0]<<std::endl;
    }
    return corner_set;
}