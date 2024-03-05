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
* @brief Creates coordinates of each corner and draws those corners in the image
* @param frame Input image
* @param boardSize size of the chessboard
* @return list of all the corners in the chessboard
*/

std::vector<cv::Point2f>  Drawchessboardcorner(cv::Mat frame, cv::Size boardSize);
#endif