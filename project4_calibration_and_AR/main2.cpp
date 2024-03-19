/**
 * author: Harshit Kumar, Khushi Neema
 * date: Mar 5th, 2024
 * purpose: Multiple target detection and augmentation
 *
 */

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "chessboardcorner.h"

using namespace std;
using namespace cv;

// Generate 3D points for the chessboard corners in the chessboard's coordinate space
std::vector<cv::Point3f> generate3DChessboardCorners(cv::Size boardSize, float squareSize) {
    std::vector<cv::Point3f> corners;
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            corners.push_back(cv::Point3f(j * squareSize, i * squareSize, 0.0f));
        }
    }
    return corners;
}


int main(int argc, char** argv) {
    cv::VideoCapture capdev(0);
    if (!capdev.isOpened()) {
        std::cerr << "Unable to open the video camera" << std::endl;
        return -1;
    }

    cv::namedWindow("Display Window", WINDOW_NORMAL);
    cv::Mat frame;

    // Assuming there could be up to 2 chessboards in the scene for this example
    std::vector<std::vector<cv::Point2f>> all_corners(2);
    cv::Size boardSize(9, 6); // Common chessboard size

    // camera
    cv::Mat camera_matrix=cv::Mat::eye(3,3, CV_64FC1);
    cv::Mat distortion_coefficients = cv::Mat::zeros(5, 1, CV_64FC1);
    cv::FileStorage fs("intrinsic_parameters.yml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: rUnable to open the file for reading." << std::endl;
        return 0;
    }
    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> distortion_coefficients;
    fs.release();

    while (true) {
        capdev >> frame;
        if (frame.empty()) {
            std::cerr << "Blank Frame grabbed" << std::endl;
            break;
        }

        for (auto& corners : all_corners) {
            corners.clear(); // Clear previous corners
        }

        // Detect and process each chessboard
        for (int i = 0; i < all_corners.size(); ++i) {
            if (drawchessboardcorner(frame, boardSize, all_corners[i])) {
                // Detected a chessboard, now calculate its pose
                cv::Mat rvec, tvec;
                calculatePose(all_corners[i], camera_matrix, distortion_coefficients, boardSize, rvec, tvec);

                // Here you would perform any augmentation for the detected chessboard
                projectPointsAndDraw(all_corners[i], rvec, tvec, camera_matrix, distortion_coefficients, boardSize, frame);
                // Blur the outside chessboard region
                // blurOutsideChessboardRegion(boardSize, rvec, tvec, camera_matrix, distortion_coefficients, frame);
                // Blend the chessboard with grass
                // cv::Mat texture = cv::imread("../grass.jpg");
                // blendChessboardRegion(boardSize, rvec, tvec, camera_matrix, distortion_coefficients, frame, texture);
                // Blend the outside chessboard with pebbles
                // cv::Mat desert = cv::imread("../desert.jpg");
                // blendOutsideChessboardRegion(boardSize, rvec, tvec, camera_matrix, distortion_coefficients, frame, desert);
                createObject(rvec, tvec, camera_matrix, distortion_coefficients, boardSize, frame);
            }
        }

        cv::imshow("Display Window", frame);
        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        }
    }

    capdev.release();
    cv::destroyAllWindows();
    return 0;
}
