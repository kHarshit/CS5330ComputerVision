/**
 * author: Harshit Kumar, Khushi Neema
 * date: Mar 5th, 2024
 * purpose: Finding and drawing chessboard corners
 *
 */

#ifndef CHESSBOARDCORNER_H
#define CHESSBOARDCORNER_H

#include <opencv2/opencv.hpp>



/**
 * @brief Draws the chessboard corners on the frame
 * @param frame input frame
 * @param boardSize size of the chessboard
 * @param corner_set list of all the corners in the chessboard
 * @return true if the corners are found, false otherwise
*/
bool drawchessboardcorner(cv::Mat frame, cv::Size boardSize, std::vector<cv::Point2f> &corner_set);

/**
 * @brief Calibrates the camera and saves the parameters
 * @param point_list list of all the points in the chessboard
 * @param corner_list list of all the corners in the chessboard
 * @param frame_size size of the frame
 * @param camera_matrix camera matrix
 * @param distortion_coefficients distortion coefficients
 * @return void
*/
void calibrateCameraAndSaveParameters(std::vector<std::vector<cv::Vec3f>>& point_list, std::vector<std::vector<cv::Point2f>>& corner_list, cv::Size frame_size, cv::Mat& camera_matrix, cv::Mat& distortion_coefficients);

/**
 * @brief Calculates the pose of the chessboard
 * @param corner_set list of all the corners in the chessboard
 * @param camera_matrix camera matrix
 * @param distortion_coefficients distortion coefficients
 * @param boardSize size of the chessboard
*/
void calculatePose(const std::vector<cv::Point2f>& corner_set, const cv::Mat& camera_matrix, const cv::Mat& distortion_coefficients, const cv::Size& boardSize);
#endif