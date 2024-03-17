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
 * @brief Calculates the pose of the chessboard
 * @param corner_set list of all the corners in the chessboard
 * @param camera_matrix camera matrix
 * @param distortion_coefficients distortion coefficients
 * @param boardSize size of the chessboard
*/
void calculatePose(const std::vector<cv::Point2f>& corner_set, const cv::Mat& camera_matrix, const cv::Mat& distortion_coefficients, const cv::Size& boardSize);
#endif